# EXP-016 : rapport de l'analyse pré-enregistrée (version 2 ; épisodes cassés par la compression)

Commit de l'analyse : `c0f24ff`. Contrôles négatifs acceptés : pusht oui, tworoom oui.

Lecture : un rejet dit « plus d'épisodes de la strate jamais réussis qu'une perte proportionnelle ne le prévoit » ; il ne sépare pas des épisodes précis d'une perte concentrée sur des épisodes fragiles ; les verdicts ne se comparent pas entre versions de pertes différentes. Excès biaisé vers le bas ; crochets : fluctuation sous le nul.

| Tâche, version | Strate (k_R = 5) | Réussite dans la strate | Jamais réussis (T1) | Attendu | Excès [fluctuation] | p | p Holm | Calibration synthétique (rejets à 5 % ; 5 %/7) | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| pusht | RTN 4 bits | 347 | 80.7 % | 22 | 0.28 | +21.7 [+20 ; +22] | 0.0001 | 0.0007 | 3.1 % ; 0.3 % (acceptée) | soutenue |
| pusht | RTN 3 bits | 347 | 43.5 % | 129 | 20.48 | +108.5 [+99 ; +117] | 0.0001 | 0.0007 | 1.9 % ; 0.0 % (acceptée) | soutenue |
| pusht | GPTQ 3 bits | 347 | 87.0 % | 15 | 0.14 | +14.9 [+14 ; +15] | 0.0001 | 0.0007 | 0.5 % ; 0.3 % (acceptée) | soutenue |
| pusht | int4 par canal, Neural Engine | 347 | 87.2 % | 15 | 0.14 | +14.9 [+14 ; +15] | 0.0001 | 0.0007 | 0.5 % ; 0.2 % (acceptée) | soutenue ; codage (excès moins 16 bits +13.9) |
| pusht | palettes, Neural Engine | 347 | 93.8 % | 3 | 0.08 | +2.9 [+2 ; +3] | 0.0001 | 0.0007 | 0.0 % ; 0.0 % (acceptée) | excès sous le minimum ; codage et moteur, ou non séparés (excès moins 16 bits +2.0) |
| tworoom | RTN 4 bits | 369 | 89.5 % | 13 | 0.08 | +12.9 [+12 ; +13] | 0.0001 | 0.0007 | 0.1 % ; 0.1 % (acceptée) | soutenue |
| tworoom | RTN 3 bits | 369 | 46.7 % | 129 | 16.24 | +112.8 [+105 ; +120] | 0.0001 | 0.0007 | 1.2 % ; 0.1 % (acceptée) | soutenue |

Contrôles :

- controles_negatifs pusht : T1 0, attendu 0.06, excès -0.1, p 1.0000 ; k_V dans la strate observé [0, 1, 2, 4, 19, 321], attendu [0.1, 0.4, 1.5, 4.8, 21.1, 319.1]
- controles_negatifs tworoom : T1 0, attendu 0.03, excès -0.0, p 1.0000 ; k_V dans la strate observé [0, 0, 1, 2, 14, 352], attendu [0.0, 0.2, 0.7, 2.8, 12.4, 352.9]
- controle_moteur pusht | 16 bits, Neural Engine : T1 1, attendu 0.05, excès +1.0, p 0.0462 ; k_V dans la strate observé [1, 0, 0, 2, 26, 318], attendu [0.0, 0.4, 1.4, 4.6, 19.8, 320.7]
- controle_moteur tworoom | 16 bits, Neural Engine : T1 0, attendu 0.03, excès -0.0, p 1.0000 ; k_V dans la strate observé [0, 0, 1, 4, 13, 351], attendu [0.0, 0.2, 0.8, 3.4, 13.9, 350.7]
- huit_bits pusht | RTN 8 bits : T1 1, attendu 0.05, excès +0.9, p 0.0506 ; k_V dans la strate observé [1, 0, 0, 6, 21, 319], attendu [0.1, 0.4, 1.5, 4.9, 21.9, 318.3]
- huit_bits tworoom | RTN 8 bits : T1 0, attendu 0.04, excès -0.0, p 1.0000 ; k_V dans la strate observé [0, 0, 0, 6, 16, 347], attendu [0.0, 0.2, 0.9, 4.0, 16.2, 347.6]

E2 (jamais réussis par valeur de k_R, observé contre attendu sous une perte proportionnelle globale) et E3 :

- pusht | RTN 4 bits : observé [22, 8, 6, 7, 4, 22], attendu [24.2, 5.9, 3.3, 1.2, 0.7, 0.3] ; excès à k_R = 5 +21.7, à k_R de 1 à 4 +14.0 ; perte moyenne 14.1 points, au meilleur des 5 6.4
- pusht | RTN 3 bits : observé [28, 11, 14, 18, 29, 129], attendu [27.4, 8.4, 8.9, 6.3, 6.9, 23.9] ; excès à k_R = 5 +105.1, à k_R de 1 à 4 +41.6 ; perte moyenne 47.8 points, au meilleur des 5 38.4
- pusht | GPTQ 3 bits : observé [18, 5, 9, 6, 5, 15], attendu [23.7, 5.6, 2.9, 0.9, 0.4, 0.1] ; excès à k_R = 5 +14.9, à k_R de 1 à 4 +15.2 ; perte moyenne 8.3 points, au meilleur des 5 4.2
- pusht | int4 par canal, Neural Engine : observé [20, 7, 5, 5, 5, 15], attendu [23.7, 5.6, 2.9, 0.9, 0.4, 0.1] ; excès à k_R = 5 +14.9, à k_R de 1 à 4 +12.1 ; perte moyenne 8.6 points, au meilleur des 5 4.0
- pusht | palettes, Neural Engine : observé [25, 6, 4, 2, 4, 3], attendu [23.2, 5.3, 2.5, 0.6, 0.3, 0.1] ; excès à k_R = 5 +2.9, à k_R de 1 à 4 +7.2 ; perte moyenne 2.4 points, au meilleur des 5 1.4
- tworoom | RTN 4 bits : observé [42, 10, 5, 3, 6, 13], attendu [40.9, 4.2, 1.3, 0.9, 0.4, 0.1] ; excès à k_R = 5 +12.9, à k_R de 1 à 4 +17.2 ; perte moyenne 11.0 points, au meilleur des 5 7.8
- tworoom | RTN 3 bits : observé [34, 7, 10, 15, 25, 129], attendu [42.8, 6.7, 4.0, 4.5, 4.0, 18.3] ; excès à k_R = 5 +110.7, à k_R de 1 à 4 +37.8 ; perte moyenne 46.2 points, au meilleur des 5 36.0
- repère pusht, pleine précision 42 à 46 : observé [24, 8, 4, 1, 0, 0], attendu [23.0, 5.2, 2.4, 0.6, 0.2, 0.1]
- repère tworoom, pleine précision 42 à 46 : observé [39, 1, 0, 0, 0, 0], attendu [40.3, 3.6, 0.9, 0.5, 0.2, 0.0]

E4 (PushT, strate) :

- RTN 4 bits et RTN 3 bits : Jaccard 0.079 (nul 0.002), 11 communs, tailles [22, 129]
- RTN 3 bits et GPTQ 3 bits : Jaccard 0.067 (nul 0.001), 9 communs, tailles [129, 15]
- RTN 4 bits et int4 par canal, Neural Engine : Jaccard 0.276 (nul 0.004), 8 communs, tailles [22, 15]
- int4 par canal, Neural Engine et palettes, Neural Engine : Jaccard 0.125 (nul 0.002), 2 communs, tailles [15, 3]

Secondaires :

- pusht | exp003/all-5 : réussite 80.5 %, T1 2 sur 347, attendu 0.08, excès +1.9, p 0.0029
- pusht | exp003/all-6 : réussite 83.2 %, T1 0 sur 347, attendu 0.06, excès -0.1, p 1.0000
- pusht | exp003/encoder-3 : réussite 53.7 %, T1 63 sur 347, attendu 2.19, excès +60.8, p 0.0001
- pusht | exp003/encoder-4 : réussite 78.9 %, T1 3 sur 347, attendu 0.09, excès +2.9, p 0.0001
- pusht | exp003/predictor-3 : réussite 81.3 %, T1 7 sur 347, attendu 0.08, excès +6.9, p 0.0001
- pusht | exp003/predictor-4 : réussite 84.2 %, T1 1 sur 347, attendu 0.06, excès +0.9, p 0.0598
- pusht | exp004/all-3-g64 : réussite 45.7 %, T1 92 sur 347, attendu 8.17, excès +83.8, p 0.0001
- pusht | exp004/all-4-g64 : réussite 75.6 %, T1 13 sur 347, attendu 0.12, excès +12.9, p 0.0001
- pusht | exp004/autres-3 : réussite 54.6 %, T1 65 sur 347, attendu 2.63, excès +62.4, p 0.0001
- pusht | exp004/autres-4 : réussite 74.1 %, T1 12 sur 347, attendu 0.16, excès +11.8, p 0.0001
- pusht | exp004/encpred-4 : réussite 79.5 %, T1 6 sur 347, attendu 0.09, excès +5.9, p 0.0001
- pusht | exp009-pusht/all-4-gptq : réussite 82.0 %, T1 1 sur 347, attendu 0.07, excès +0.9, p 0.0660
- pusht | exp009-pusht/autres-4-gptq : réussite 82.8 %, T1 0 sur 347, attendu 0.06, excès -0.1, p 1.0000
- pusht | exp009-pusht/encoder-4-gptq : réussite 83.1 %, T1 2 sur 347, attendu 0.06, excès +1.9, p 0.0020
- pusht | exp011-pusht/predictor-2-clip : réussite 40.4 %, T1 112 sur 347, attendu 14.37, excès +97.6, p 0.0001
- pusht | exp011-pusht/predictor-2-clip-gptq : réussite 53.0 %, T1 77 sur 347, attendu 3.45, excès +73.6, p 0.0001
- pusht | exp014-int8-pusht/fp32 : réussite 84.1 %, T1 0 sur 347, attendu 0.06, excès -0.1, p 1.0000
- tworoom | exp006/all-4-g64 : réussite 80.3 %, T1 4 sur 369, attendu 0.05, excès +4.0, p 0.0001
- tworoom | exp006/all-6 : réussite 83.6 %, T1 0 sur 369, attendu 0.04, excès -0.0, p 1.0000
- tworoom | exp006/autres-4 : réussite 83.6 %, T1 0 sur 369, attendu 0.04, excès -0.0, p 1.0000
- tworoom | exp006/encoder-4 : réussite 75.7 %, T1 14 sur 369, attendu 0.07, excès +13.9, p 0.0001
- tworoom | exp006/predictor-4 : réussite 84.9 %, T1 0 sur 369, attendu 0.04, excès -0.0, p 1.0000
- tworoom | exp008/all-3-g64 : réussite 37.9 %, T1 147 sur 369, attendu 16.79, excès +130.2, p 0.0001
- tworoom | exp008/all-5 : réussite 83.5 %, T1 1 sur 369, attendu 0.04, excès +1.0, p 0.0404
- tworoom | exp008/autres-3 : réussite 82.7 %, T1 5 sur 369, attendu 0.04, excès +5.0, p 0.0001
- tworoom | exp008/encoder-3 : réussite 47.0 %, T1 111 sur 369, attendu 5.84, excès +105.2, p 0.0001
- tworoom | exp008/encpred-4 : réussite 75.6 %, T1 12 sur 369, attendu 0.06, excès +11.9, p 0.0001
- tworoom | exp008/predictor-3 : réussite 83.8 %, T1 0 sur 369, attendu 0.04, excès -0.0, p 1.0000
- tworoom | exp009-tworoom/all-3-gptq : réussite 83.9 %, T1 0 sur 369, attendu 0.04, excès -0.0, p 1.0000
- tworoom | exp009-tworoom/all-4-gptq : réussite 84.6 %, T1 0 sur 369, attendu 0.04, excès -0.0, p 1.0000
- tworoom | exp009-tworoom/autres-4-gptq : réussite 84.8 %, T1 0 sur 369, attendu 0.03, excès -0.0, p 1.0000
- tworoom | exp009-tworoom/encoder-4-gptq : réussite 84.9 %, T1 0 sur 369, attendu 0.04, excès -0.0, p 1.0000
- tworoom | exp011-tworoom/predictor-2-clip : réussite 84.5 %, T1 0 sur 369, attendu 0.04, excès -0.0, p 1.0000
- tworoom | exp011-tworoom/predictor-2-clip-gptq : réussite 81.4 %, T1 0 sur 369, attendu 0.04, excès -0.0, p 1.0000
- tworoom | exp014-int8-tworoom/fp32 : réussite 85.1 %, T1 0 sur 369, attendu 0.03, excès -0.0, p 1.0000
- tworoom | exp014-int4c-tworoom/fp32 : réussite 84.2 %, T1 0 sur 369, attendu 0.03, excès -0.0, p 1.0000
- tworoom | exp014-pal4-tworoom/fp32 : réussite 85.2 %, T1 0 sur 369, attendu 0.03, excès -0.0, p 1.0000

Loi a priori (grille, EM sur 10 graines) et bêta-binomiale pour comparaison :

- pusht : poids non nuls {'0.025': 0.003976362468276656, '0.030': 0.011406946863716693, '0.035': 0.017049203323232672, '0.040': 0.015388678829295423, '0.045': 0.009527009267475826, '0.050': 0.004511303181414094, '0.055': 0.0017921983180105433, '0.180': 0.001039222963071622, '0.185': 0.0016472749507520197, '0.190': 0.0024259455793904975, '0.195': 0.0032913622279400097, '0.200': 0.004084571587877673, '0.205': 0.004609668234168061, '0.210': 0.004709765906293156, '0.215': 0.004342703851245511, '0.220': 0.0036069069794882024, '0.225': 0.0026967866801039025, '0.230': 0.0018160899331267576, '0.235': 0.001103426959308117, '0.565': 0.001761486780935034, '0.570': 0.0030867731078059817, '0.575': 0.004986965850796593, '0.580': 0.0074145467369544135, '0.585': 0.010132175026208744, '0.590': 0.012717162815489746, '0.595': 0.014658678668683861, '0.600': 0.015524309013759466, '0.605': 0.015121173744567845, '0.610': 0.013567565399070712, '0.615': 0.011238011381493123, '0.620': 0.008616107890566061, '0.625': 0.00613420249309018, '0.630': 0.004070441875086513, '0.635': 0.0025280341473300017, '0.640': 0.001476396523746506, '0.880': 0.001595063664815036, '0.885': 0.0030089763942151466, '0.890': 0.005259831653435384, '0.895': 0.008461893898673689, '0.900': 0.012458776409252963, '0.905': 0.016718818881399915, '0.910': 0.020399424296193507, '0.915': 0.02262230949442169, '0.920': 0.022844127777583052, '0.925': 0.02109889111181201, '0.930': 0.01795455290302899, '0.935': 0.014226556128792641, '0.940': 0.010645466100007918, '0.945': 0.007659608427019673, '0.950': 0.005419212580103283, '0.955': 0.0038735600175378456, '0.960': 0.0028887546668968974, '0.965': 0.002334152085286136, '0.970': 0.0021348041307650977, '0.975': 0.002323725629660621, '0.980': 0.0031870625570772065, '0.985': 0.005874216296218349, '0.990': 0.01563982217448849, '0.995': 0.06519259879748471, '1.000': 0.46515688971154767} ; bêta-binomiale sur 10 essais : alpha 0.582, beta 0.118, p d'adéquation 0.3876
- tworoom : poids non nuls {'0.000': 0.01816800714186527, '0.005': 0.022932076518843617, '0.010': 0.019085970715058166, '0.015': 0.011622744724631946, '0.020': 0.005662157460764919, '0.025': 0.002380587915850216, '0.165': 0.0010632782672968417, '0.170': 0.001452059482407936, '0.175': 0.0018655307636187008, '0.180': 0.00224613758064935, '0.185': 0.002527288073081434, '0.190': 0.0026524639693446335, '0.195': 0.0025943130883396863, '0.200': 0.002364658375942189, '0.205': 0.0020102783665248767, '0.210': 0.0015966369587494759, '0.215': 0.0011875902745628276, '0.490': 0.0011380856215848746, '0.495': 0.0017920204458840637, '0.500': 0.0026770164841996742, '0.505': 0.0037796761680365127, '0.510': 0.005026489179157155, '0.515': 0.006277099231119614, '0.520': 0.007341600448538419, '0.525': 0.008024131644936016, '0.530': 0.008181090208412602, '0.535': 0.0077706841364442655, '0.540': 0.006870299189906576, '0.545': 0.0056520025850086125, '0.550': 0.004327061983718087, '0.555': 0.0030847177759261986, '0.560': 0.0020499878830043576, '0.565': 0.0012720280426082066, '0.850': 0.001036554578804153, '0.855': 0.0023342339995997743, '0.860': 0.004720526826306106, '0.865': 0.008488393683478942, '0.870': 0.01345254947489287, '0.875': 0.018648205418047504, '0.880': 0.022474345244990968, '0.885': 0.02344665298940413, '0.890': 0.0211261652818477, '0.895': 0.016440138092901107, '0.900': 0.011077933918879415, '0.905': 0.006499257157867852, '0.910': 0.003348923357452348, '0.915': 0.00153433200951374, '0.995': 0.0071415841041455996, '1.000': 0.6509261894233588} ; bêta-binomiale sur 10 essais : alpha 0.351, beta 0.070, p d'adéquation 0.1961
