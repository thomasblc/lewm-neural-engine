"""Énergie consommée par composant de la puce Apple (CPU, GPU, Neural Engine, mémoire...), lue dans les compteurs
IOReport du groupe « Energy Model », sans droits administrateur (la même source que l'app Stats ou macmon ; powermetrics
demande sudo). Bibliothèque système chargée par ctypes, rien à installer. API privée d'Apple : à valider contre une
mesure indépendante avant de s'y fier.

Usage :
    python3 outils/energie_puce.py 5          # énergie et puissance moyenne par canal sur 5 s

Dans un script :
    from energie_puce import Compteur
    c = Compteur(); e0 = c.lire(); ... ; e1 = c.lire(); Compteur.ecart(e0, e1)   # {canal: joules}
"""
import ctypes
import sys
import time

_cf = ctypes.CDLL("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
_io = ctypes.CDLL("/usr/lib/libIOReport.dylib")      # dans le cache partagé du système
_vp, _u64, _i64 = ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int64
UTF8 = 0x08000100


def _sig(f, res, *args):
    f.restype, f.argtypes = res, list(args)


_sig(_cf.CFStringCreateWithCString, _vp, _vp, ctypes.c_char_p, ctypes.c_uint32)
_sig(_cf.CFStringGetCString, ctypes.c_bool, _vp, ctypes.c_char_p, ctypes.c_long, ctypes.c_uint32)
_sig(_cf.CFDictionaryGetValue, _vp, _vp, _vp)
_sig(_cf.CFArrayGetCount, ctypes.c_long, _vp)
_sig(_cf.CFArrayGetValueAtIndex, _vp, _vp, ctypes.c_long)
_sig(_cf.CFRelease, None, _vp)
_sig(_io.IOReportCopyChannelsInGroup, _vp, _vp, _vp, _u64, _u64, _u64)
_sig(_io.IOReportCreateSubscription, _vp, _vp, _vp, ctypes.POINTER(_vp), _u64, _vp)
_sig(_io.IOReportCreateSamples, _vp, _vp, _vp, _vp)
_sig(_io.IOReportCreateSamplesDelta, _vp, _vp, _vp, _vp)
_sig(_io.IOReportChannelGetChannelName, _vp, _vp)
_sig(_io.IOReportChannelGetUnitLabel, _vp, _vp)
_sig(_io.IOReportSimpleGetIntegerValue, _i64, _vp, _vp)


def _cfs(s):
    return _cf.CFStringCreateWithCString(None, s.encode(), UTF8)


def _py(cfstr):
    if not cfstr:
        return ""
    b = ctypes.create_string_buffer(256)
    return b.value.decode() if _cf.CFStringGetCString(cfstr, b, 256, UTF8) else ""


_UNITE = {"mJ": 1e-3, "uJ": 1e-6, "µJ": 1e-6, "nJ": 1e-9}


class Compteur:
    """Abonnement aux canaux « Energy Model ». lire() rend un échantillon brut (cumuls) ; ecart() les joules entre deux."""

    def __init__(self):
        canaux = _io.IOReportCopyChannelsInGroup(_cfs("Energy Model"), None, 0, 0, 0)
        if not canaux:
            raise RuntimeError("canaux IOReport « Energy Model » indisponibles")
        self._abonnes = _vp()
        self._sub = _io.IOReportCreateSubscription(None, canaux, ctypes.byref(self._abonnes), 0, None)
        if not self._sub:
            raise RuntimeError("abonnement IOReport refusé")

    def lire(self):
        return (time.time(), _io.IOReportCreateSamples(self._sub, self._abonnes, None))

    @staticmethod
    def ecart(e0, e1):
        """Joules par canal entre deux échantillons, et durée (s)."""
        (t0, s0), (t1, s1) = e0, e1
        d = _io.IOReportCreateSamplesDelta(s0, s1, None)
        tab = _cf.CFDictionaryGetValue(d, _cfs("IOReportChannels"))
        res = {}
        for i in range(_cf.CFArrayGetCount(tab)):
            it = _cf.CFArrayGetValueAtIndex(tab, i)
            nom, unite = _py(_io.IOReportChannelGetChannelName(it)), _py(_io.IOReportChannelGetUnitLabel(it))
            v = _io.IOReportSimpleGetIntegerValue(it, None)
            res[nom] = res.get(nom, 0.0) + v * _UNITE.get(unite.strip(), float("nan"))
        _cf.CFRelease(d)
        return res, t1 - t0


if __name__ == "__main__":
    duree = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0
    c = Compteur()
    e0 = c.lire()
    time.sleep(duree)
    j, dt = Compteur.ecart(e0, c.lire())
    for nom, v in sorted(j.items(), key=lambda x: -abs(x[1]) if x[1] == x[1] else 0):
        if v != 0:
            print(f"{nom:28s} {v:9.3f} J  {v / dt:8.3f} W")
