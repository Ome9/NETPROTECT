# Network Anomaly Detection Evaluation Report

## Reconstruction Error Statistics

- Mean: 2.205628
- Standard Deviation: 68.021034
- Minimum: 0.001546
- Maximum: 3101.936768
- Median: 0.288859

## Threshold-based Evaluation Results

### Percentile 90

- Threshold: 0.277036
- Accuracy: 0.8509
- Precision: 0.9149
- Recall: 0.8137
- F1-Score: 0.8613
- ROC-AUC: 0.9213
- True Positives: 10442
- False Positives: 971
- True Negatives: 8740
- False Negatives: 2391

#### Per-Attack Type Performance:

**apache2** (n=737)
- Accuracy: 0.8928
- Precision: 1.0000
- Recall: 0.8928
- F1-Score: 0.9434

**back** (n=359)
- Accuracy: 0.0641
- Precision: 1.0000
- Recall: 0.0641
- F1-Score: 0.1204

**buffer_overflow** (n=20)
- Accuracy: 0.8500
- Precision: 1.0000
- Recall: 0.8500
- F1-Score: 0.9189

**ftp_write** (n=3)
- Accuracy: 0.6667
- Precision: 1.0000
- Recall: 0.6667
- F1-Score: 0.8000

**guess_passwd** (n=1231)
- Accuracy: 0.4159
- Precision: 1.0000
- Recall: 0.4159
- F1-Score: 0.5875

**httptunnel** (n=133)
- Accuracy: 0.9248
- Precision: 1.0000
- Recall: 0.9248
- F1-Score: 0.9609

**imap** (n=1)
- Accuracy: 1.0000
- Precision: 1.0000
- Recall: 1.0000
- F1-Score: 1.0000

**ipsweep** (n=141)
- Accuracy: 1.0000
- Precision: 1.0000
- Recall: 1.0000
- F1-Score: 1.0000

**land** (n=7)
- Accuracy: 1.0000
- Precision: 1.0000
- Recall: 1.0000
- F1-Score: 1.0000

**loadmodule** (n=2)
- Accuracy: 1.0000
- Precision: 1.0000
- Recall: 1.0000
- F1-Score: 1.0000

**mailbomb** (n=293)
- Accuracy: 0.1331
- Precision: 1.0000
- Recall: 0.1331
- F1-Score: 0.2349

**mscan** (n=996)
- Accuracy: 0.9880
- Precision: 1.0000
- Recall: 0.9880
- F1-Score: 0.9939

**multihop** (n=18)
- Accuracy: 0.4444
- Precision: 1.0000
- Recall: 0.4444
- F1-Score: 0.6154

**named** (n=17)
- Accuracy: 0.3529
- Precision: 1.0000
- Recall: 0.3529
- F1-Score: 0.5217

**neptune** (n=4657)
- Accuracy: 1.0000
- Precision: 1.0000
- Recall: 1.0000
- F1-Score: 1.0000

**nmap** (n=73)
- Accuracy: 1.0000
- Precision: 1.0000
- Recall: 1.0000
- F1-Score: 1.0000

**normal** (n=9711)
- Accuracy: 0.9000
- Precision: 0.0000
- Recall: 0.0000
- F1-Score: 0.0000

**perl** (n=2)
- Accuracy: 1.0000
- Precision: 1.0000
- Recall: 1.0000
- F1-Score: 1.0000

**phf** (n=2)
- Accuracy: 1.0000
- Precision: 1.0000
- Recall: 1.0000
- F1-Score: 1.0000

**pod** (n=41)
- Accuracy: 0.9756
- Precision: 1.0000
- Recall: 0.9756
- F1-Score: 0.9877

**portsweep** (n=157)
- Accuracy: 1.0000
- Precision: 1.0000
- Recall: 1.0000
- F1-Score: 1.0000

**processtable** (n=685)
- Accuracy: 1.0000
- Precision: 1.0000
- Recall: 1.0000
- F1-Score: 1.0000

**ps** (n=15)
- Accuracy: 0.4667
- Precision: 1.0000
- Recall: 0.4667
- F1-Score: 0.6364

**rootkit** (n=13)
- Accuracy: 0.2308
- Precision: 1.0000
- Recall: 0.2308
- F1-Score: 0.3750

**saint** (n=319)
- Accuracy: 0.9530
- Precision: 1.0000
- Recall: 0.9530
- F1-Score: 0.9759

**satan** (n=735)
- Accuracy: 0.9469
- Precision: 1.0000
- Recall: 0.9469
- F1-Score: 0.9727

**sendmail** (n=14)
- Accuracy: 0.7857
- Precision: 1.0000
- Recall: 0.7857
- F1-Score: 0.8800

**smurf** (n=665)
- Accuracy: 1.0000
- Precision: 1.0000
- Recall: 1.0000
- F1-Score: 1.0000

**snmpgetattack** (n=178)
- Accuracy: 0.0899
- Precision: 1.0000
- Recall: 0.0899
- F1-Score: 0.1649

**snmpguess** (n=331)
- Accuracy: 0.0181
- Precision: 1.0000
- Recall: 0.0181
- F1-Score: 0.0356

**sqlattack** (n=2)
- Accuracy: 1.0000
- Precision: 1.0000
- Recall: 1.0000
- F1-Score: 1.0000

**teardrop** (n=12)
- Accuracy: 1.0000
- Precision: 1.0000
- Recall: 1.0000
- F1-Score: 1.0000

**udpstorm** (n=2)
- Accuracy: 0.0000
- Precision: 0.0000
- Recall: 0.0000
- F1-Score: 0.0000

**warezmaster** (n=944)
- Accuracy: 0.5932
- Precision: 1.0000
- Recall: 0.5932
- F1-Score: 0.7447

**worm** (n=2)
- Accuracy: 0.0000
- Precision: 0.0000
- Recall: 0.0000
- F1-Score: 0.0000

**xlock** (n=9)
- Accuracy: 0.6667
- Precision: 1.0000
- Recall: 0.6667
- F1-Score: 0.8000

**xsnoop** (n=4)
- Accuracy: 0.7500
- Precision: 1.0000
- Recall: 0.7500
- F1-Score: 0.8571

**xterm** (n=13)
- Accuracy: 0.9231
- Precision: 1.0000
- Recall: 0.9231
- F1-Score: 0.9600


### Percentile 95

- Threshold: 0.790794
- Accuracy: 0.5653
- Precision: 0.8787
- Recall: 0.2743
- F1-Score: 0.4181
- ROC-AUC: 0.9213
- True Positives: 3520
- False Positives: 486
- True Negatives: 9225
- False Negatives: 9313

#### Per-Attack Type Performance:

**apache2** (n=737)
- Accuracy: 0.0000
- Precision: 0.0000
- Recall: 0.0000
- F1-Score: 0.0000

**back** (n=359)
- Accuracy: 0.0000
- Precision: 0.0000
- Recall: 0.0000
- F1-Score: 0.0000

**buffer_overflow** (n=20)
- Accuracy: 0.8000
- Precision: 1.0000
- Recall: 0.8000
- F1-Score: 0.8889

**ftp_write** (n=3)
- Accuracy: 0.6667
- Precision: 1.0000
- Recall: 0.6667
- F1-Score: 0.8000

**guess_passwd** (n=1231)
- Accuracy: 0.3802
- Precision: 1.0000
- Recall: 0.3802
- F1-Score: 0.5509

**httptunnel** (n=133)
- Accuracy: 0.1278
- Precision: 1.0000
- Recall: 0.1278
- F1-Score: 0.2267

**imap** (n=1)
- Accuracy: 0.0000
- Precision: 0.0000
- Recall: 0.0000
- F1-Score: 0.0000

**ipsweep** (n=141)
- Accuracy: 0.9645
- Precision: 1.0000
- Recall: 0.9645
- F1-Score: 0.9819

**land** (n=7)
- Accuracy: 1.0000
- Precision: 1.0000
- Recall: 1.0000
- F1-Score: 1.0000

**loadmodule** (n=2)
- Accuracy: 1.0000
- Precision: 1.0000
- Recall: 1.0000
- F1-Score: 1.0000

**mailbomb** (n=293)
- Accuracy: 0.0000
- Precision: 0.0000
- Recall: 0.0000
- F1-Score: 0.0000

**mscan** (n=996)
- Accuracy: 0.5522
- Precision: 1.0000
- Recall: 0.5522
- F1-Score: 0.7115

**multihop** (n=18)
- Accuracy: 0.3889
- Precision: 1.0000
- Recall: 0.3889
- F1-Score: 0.5600

**named** (n=17)
- Accuracy: 0.1765
- Precision: 1.0000
- Recall: 0.1765
- F1-Score: 0.3000

**neptune** (n=4657)
- Accuracy: 0.1102
- Precision: 1.0000
- Recall: 0.1102
- F1-Score: 0.1985

**nmap** (n=73)
- Accuracy: 1.0000
- Precision: 1.0000
- Recall: 1.0000
- F1-Score: 1.0000

**normal** (n=9711)
- Accuracy: 0.9500
- Precision: 0.0000
- Recall: 0.0000
- F1-Score: 0.0000

**perl** (n=2)
- Accuracy: 1.0000
- Precision: 1.0000
- Recall: 1.0000
- F1-Score: 1.0000

**phf** (n=2)
- Accuracy: 0.5000
- Precision: 1.0000
- Recall: 0.5000
- F1-Score: 0.6667

**pod** (n=41)
- Accuracy: 0.8537
- Precision: 1.0000
- Recall: 0.8537
- F1-Score: 0.9211

**portsweep** (n=157)
- Accuracy: 0.6688
- Precision: 1.0000
- Recall: 0.6688
- F1-Score: 0.8015

**processtable** (n=685)
- Accuracy: 0.0321
- Precision: 1.0000
- Recall: 0.0321
- F1-Score: 0.0622

**ps** (n=15)
- Accuracy: 0.4000
- Precision: 1.0000
- Recall: 0.4000
- F1-Score: 0.5714

**rootkit** (n=13)
- Accuracy: 0.2308
- Precision: 1.0000
- Recall: 0.2308
- F1-Score: 0.3750

**saint** (n=319)
- Accuracy: 0.6270
- Precision: 1.0000
- Recall: 0.6270
- F1-Score: 0.7707

**satan** (n=735)
- Accuracy: 0.6435
- Precision: 1.0000
- Recall: 0.6435
- F1-Score: 0.7831

**sendmail** (n=14)
- Accuracy: 0.3571
- Precision: 1.0000
- Recall: 0.3571
- F1-Score: 0.5263

**smurf** (n=665)
- Accuracy: 0.5188
- Precision: 1.0000
- Recall: 0.5188
- F1-Score: 0.6832

**snmpgetattack** (n=178)
- Accuracy: 0.0000
- Precision: 0.0000
- Recall: 0.0000
- F1-Score: 0.0000

**snmpguess** (n=331)
- Accuracy: 0.0000
- Precision: 0.0000
- Recall: 0.0000
- F1-Score: 0.0000

**sqlattack** (n=2)
- Accuracy: 1.0000
- Precision: 1.0000
- Recall: 1.0000
- F1-Score: 1.0000

**teardrop** (n=12)
- Accuracy: 0.6667
- Precision: 1.0000
- Recall: 0.6667
- F1-Score: 0.8000

**udpstorm** (n=2)
- Accuracy: 0.0000
- Precision: 0.0000
- Recall: 0.0000
- F1-Score: 0.0000

**warezmaster** (n=944)
- Accuracy: 0.5339
- Precision: 1.0000
- Recall: 0.5339
- F1-Score: 0.6961

**worm** (n=2)
- Accuracy: 0.0000
- Precision: 0.0000
- Recall: 0.0000
- F1-Score: 0.0000

**xlock** (n=9)
- Accuracy: 0.3333
- Precision: 1.0000
- Recall: 0.3333
- F1-Score: 0.5000

**xsnoop** (n=4)
- Accuracy: 0.5000
- Precision: 1.0000
- Recall: 0.5000
- F1-Score: 0.6667

**xterm** (n=13)
- Accuracy: 0.7692
- Precision: 1.0000
- Recall: 0.7692
- F1-Score: 0.8696


### Percentile 98

- Threshold: 0.885488
- Accuracy: 0.5506
- Precision: 0.9375
- Recall: 0.2255
- F1-Score: 0.3636
- ROC-AUC: 0.9213
- True Positives: 2894
- False Positives: 193
- True Negatives: 9518
- False Negatives: 9939

#### Per-Attack Type Performance:

**apache2** (n=737)
- Accuracy: 0.0000
- Precision: 0.0000
- Recall: 0.0000
- F1-Score: 0.0000

**back** (n=359)
- Accuracy: 0.0000
- Precision: 0.0000
- Recall: 0.0000
- F1-Score: 0.0000

**buffer_overflow** (n=20)
- Accuracy: 0.8000
- Precision: 1.0000
- Recall: 0.8000
- F1-Score: 0.8889

**ftp_write** (n=3)
- Accuracy: 0.6667
- Precision: 1.0000
- Recall: 0.6667
- F1-Score: 0.8000

**guess_passwd** (n=1231)
- Accuracy: 0.3802
- Precision: 1.0000
- Recall: 0.3802
- F1-Score: 0.5509

**httptunnel** (n=133)
- Accuracy: 0.1203
- Precision: 1.0000
- Recall: 0.1203
- F1-Score: 0.2148

**imap** (n=1)
- Accuracy: 0.0000
- Precision: 0.0000
- Recall: 0.0000
- F1-Score: 0.0000

**ipsweep** (n=141)
- Accuracy: 0.9645
- Precision: 1.0000
- Recall: 0.9645
- F1-Score: 0.9819

**land** (n=7)
- Accuracy: 1.0000
- Precision: 1.0000
- Recall: 1.0000
- F1-Score: 1.0000

**loadmodule** (n=2)
- Accuracy: 1.0000
- Precision: 1.0000
- Recall: 1.0000
- F1-Score: 1.0000

**mailbomb** (n=293)
- Accuracy: 0.0000
- Precision: 0.0000
- Recall: 0.0000
- F1-Score: 0.0000

**mscan** (n=996)
- Accuracy: 0.5301
- Precision: 1.0000
- Recall: 0.5301
- F1-Score: 0.6929

**multihop** (n=18)
- Accuracy: 0.3889
- Precision: 1.0000
- Recall: 0.3889
- F1-Score: 0.5600

**named** (n=17)
- Accuracy: 0.1765
- Precision: 1.0000
- Recall: 0.1765
- F1-Score: 0.3000

**neptune** (n=4657)
- Accuracy: 0.0099
- Precision: 1.0000
- Recall: 0.0099
- F1-Score: 0.0196

**nmap** (n=73)
- Accuracy: 0.5616
- Precision: 1.0000
- Recall: 0.5616
- F1-Score: 0.7193

**normal** (n=9711)
- Accuracy: 0.9801
- Precision: 0.0000
- Recall: 0.0000
- F1-Score: 0.0000

**perl** (n=2)
- Accuracy: 1.0000
- Precision: 1.0000
- Recall: 1.0000
- F1-Score: 1.0000

**phf** (n=2)
- Accuracy: 0.5000
- Precision: 1.0000
- Recall: 0.5000
- F1-Score: 0.6667

**pod** (n=41)
- Accuracy: 0.7317
- Precision: 1.0000
- Recall: 0.7317
- F1-Score: 0.8451

**portsweep** (n=157)
- Accuracy: 0.5669
- Precision: 1.0000
- Recall: 0.5669
- F1-Score: 0.7236

**processtable** (n=685)
- Accuracy: 0.0015
- Precision: 1.0000
- Recall: 0.0015
- F1-Score: 0.0029

**ps** (n=15)
- Accuracy: 0.4000
- Precision: 1.0000
- Recall: 0.4000
- F1-Score: 0.5714

**rootkit** (n=13)
- Accuracy: 0.2308
- Precision: 1.0000
- Recall: 0.2308
- F1-Score: 0.3750

**saint** (n=319)
- Accuracy: 0.6176
- Precision: 1.0000
- Recall: 0.6176
- F1-Score: 0.7636

**satan** (n=735)
- Accuracy: 0.6313
- Precision: 1.0000
- Recall: 0.6313
- F1-Score: 0.7740

**sendmail** (n=14)
- Accuracy: 0.3571
- Precision: 1.0000
- Recall: 0.3571
- F1-Score: 0.5263

**smurf** (n=665)
- Accuracy: 0.4436
- Precision: 1.0000
- Recall: 0.4436
- F1-Score: 0.6146

**snmpgetattack** (n=178)
- Accuracy: 0.0000
- Precision: 0.0000
- Recall: 0.0000
- F1-Score: 0.0000

**snmpguess** (n=331)
- Accuracy: 0.0000
- Precision: 0.0000
- Recall: 0.0000
- F1-Score: 0.0000

**sqlattack** (n=2)
- Accuracy: 1.0000
- Precision: 1.0000
- Recall: 1.0000
- F1-Score: 1.0000

**teardrop** (n=12)
- Accuracy: 0.6667
- Precision: 1.0000
- Recall: 0.6667
- F1-Score: 0.8000

**udpstorm** (n=2)
- Accuracy: 0.0000
- Precision: 0.0000
- Recall: 0.0000
- F1-Score: 0.0000

**warezmaster** (n=944)
- Accuracy: 0.5339
- Precision: 1.0000
- Recall: 0.5339
- F1-Score: 0.6961

**worm** (n=2)
- Accuracy: 0.0000
- Precision: 0.0000
- Recall: 0.0000
- F1-Score: 0.0000

**xlock** (n=9)
- Accuracy: 0.3333
- Precision: 1.0000
- Recall: 0.3333
- F1-Score: 0.5000

**xsnoop** (n=4)
- Accuracy: 0.5000
- Precision: 1.0000
- Recall: 0.5000
- F1-Score: 0.6667

**xterm** (n=13)
- Accuracy: 0.7692
- Precision: 1.0000
- Recall: 0.7692
- F1-Score: 0.8696


### Optimal F1

- Threshold: 0.001546
- Accuracy: 0.5693
- Precision: 0.5693
- Recall: 1.0000
- F1-Score: 0.7255
- ROC-AUC: 0.9213
- True Positives: 12833
- False Positives: 9710
- True Negatives: 1
- False Negatives: 0

#### Per-Attack Type Performance:

**apache2** (n=737)
- Accuracy: 1.0000
- Precision: 1.0000
- Recall: 1.0000
- F1-Score: 1.0000

**back** (n=359)
- Accuracy: 1.0000
- Precision: 1.0000
- Recall: 1.0000
- F1-Score: 1.0000

**buffer_overflow** (n=20)
- Accuracy: 1.0000
- Precision: 1.0000
- Recall: 1.0000
- F1-Score: 1.0000

**ftp_write** (n=3)
- Accuracy: 1.0000
- Precision: 1.0000
- Recall: 1.0000
- F1-Score: 1.0000

**guess_passwd** (n=1231)
- Accuracy: 1.0000
- Precision: 1.0000
- Recall: 1.0000
- F1-Score: 1.0000

**httptunnel** (n=133)
- Accuracy: 1.0000
- Precision: 1.0000
- Recall: 1.0000
- F1-Score: 1.0000

**imap** (n=1)
- Accuracy: 1.0000
- Precision: 1.0000
- Recall: 1.0000
- F1-Score: 1.0000

**ipsweep** (n=141)
- Accuracy: 1.0000
- Precision: 1.0000
- Recall: 1.0000
- F1-Score: 1.0000

**land** (n=7)
- Accuracy: 1.0000
- Precision: 1.0000
- Recall: 1.0000
- F1-Score: 1.0000

**loadmodule** (n=2)
- Accuracy: 1.0000
- Precision: 1.0000
- Recall: 1.0000
- F1-Score: 1.0000

**mailbomb** (n=293)
- Accuracy: 1.0000
- Precision: 1.0000
- Recall: 1.0000
- F1-Score: 1.0000

**mscan** (n=996)
- Accuracy: 1.0000
- Precision: 1.0000
- Recall: 1.0000
- F1-Score: 1.0000

**multihop** (n=18)
- Accuracy: 1.0000
- Precision: 1.0000
- Recall: 1.0000
- F1-Score: 1.0000

**named** (n=17)
- Accuracy: 1.0000
- Precision: 1.0000
- Recall: 1.0000
- F1-Score: 1.0000

**neptune** (n=4657)
- Accuracy: 1.0000
- Precision: 1.0000
- Recall: 1.0000
- F1-Score: 1.0000

**nmap** (n=73)
- Accuracy: 1.0000
- Precision: 1.0000
- Recall: 1.0000
- F1-Score: 1.0000

**normal** (n=9711)
- Accuracy: 0.0001
- Precision: 0.0000
- Recall: 0.0000
- F1-Score: 0.0000

**perl** (n=2)
- Accuracy: 1.0000
- Precision: 1.0000
- Recall: 1.0000
- F1-Score: 1.0000

**phf** (n=2)
- Accuracy: 1.0000
- Precision: 1.0000
- Recall: 1.0000
- F1-Score: 1.0000

**pod** (n=41)
- Accuracy: 1.0000
- Precision: 1.0000
- Recall: 1.0000
- F1-Score: 1.0000

**portsweep** (n=157)
- Accuracy: 1.0000
- Precision: 1.0000
- Recall: 1.0000
- F1-Score: 1.0000

**processtable** (n=685)
- Accuracy: 1.0000
- Precision: 1.0000
- Recall: 1.0000
- F1-Score: 1.0000

**ps** (n=15)
- Accuracy: 1.0000
- Precision: 1.0000
- Recall: 1.0000
- F1-Score: 1.0000

**rootkit** (n=13)
- Accuracy: 1.0000
- Precision: 1.0000
- Recall: 1.0000
- F1-Score: 1.0000

**saint** (n=319)
- Accuracy: 1.0000
- Precision: 1.0000
- Recall: 1.0000
- F1-Score: 1.0000

**satan** (n=735)
- Accuracy: 1.0000
- Precision: 1.0000
- Recall: 1.0000
- F1-Score: 1.0000

**sendmail** (n=14)
- Accuracy: 1.0000
- Precision: 1.0000
- Recall: 1.0000
- F1-Score: 1.0000

**smurf** (n=665)
- Accuracy: 1.0000
- Precision: 1.0000
- Recall: 1.0000
- F1-Score: 1.0000

**snmpgetattack** (n=178)
- Accuracy: 1.0000
- Precision: 1.0000
- Recall: 1.0000
- F1-Score: 1.0000

**snmpguess** (n=331)
- Accuracy: 1.0000
- Precision: 1.0000
- Recall: 1.0000
- F1-Score: 1.0000

**sqlattack** (n=2)
- Accuracy: 1.0000
- Precision: 1.0000
- Recall: 1.0000
- F1-Score: 1.0000

**teardrop** (n=12)
- Accuracy: 1.0000
- Precision: 1.0000
- Recall: 1.0000
- F1-Score: 1.0000

**udpstorm** (n=2)
- Accuracy: 1.0000
- Precision: 1.0000
- Recall: 1.0000
- F1-Score: 1.0000

**warezmaster** (n=944)
- Accuracy: 1.0000
- Precision: 1.0000
- Recall: 1.0000
- F1-Score: 1.0000

**worm** (n=2)
- Accuracy: 1.0000
- Precision: 1.0000
- Recall: 1.0000
- F1-Score: 1.0000

**xlock** (n=9)
- Accuracy: 1.0000
- Precision: 1.0000
- Recall: 1.0000
- F1-Score: 1.0000

**xsnoop** (n=4)
- Accuracy: 1.0000
- Precision: 1.0000
- Recall: 1.0000
- F1-Score: 1.0000

**xterm** (n=13)
- Accuracy: 1.0000
- Precision: 1.0000
- Recall: 1.0000
- F1-Score: 1.0000


### Optimal Recall

- Threshold: 0.001546
- Accuracy: 0.5693
- Precision: 0.5693
- Recall: 1.0000
- F1-Score: 0.7255
- ROC-AUC: 0.9213
- True Positives: 12833
- False Positives: 9710
- True Negatives: 1
- False Negatives: 0

#### Per-Attack Type Performance:

**apache2** (n=737)
- Accuracy: 1.0000
- Precision: 1.0000
- Recall: 1.0000
- F1-Score: 1.0000

**back** (n=359)
- Accuracy: 1.0000
- Precision: 1.0000
- Recall: 1.0000
- F1-Score: 1.0000

**buffer_overflow** (n=20)
- Accuracy: 1.0000
- Precision: 1.0000
- Recall: 1.0000
- F1-Score: 1.0000

**ftp_write** (n=3)
- Accuracy: 1.0000
- Precision: 1.0000
- Recall: 1.0000
- F1-Score: 1.0000

**guess_passwd** (n=1231)
- Accuracy: 1.0000
- Precision: 1.0000
- Recall: 1.0000
- F1-Score: 1.0000

**httptunnel** (n=133)
- Accuracy: 1.0000
- Precision: 1.0000
- Recall: 1.0000
- F1-Score: 1.0000

**imap** (n=1)
- Accuracy: 1.0000
- Precision: 1.0000
- Recall: 1.0000
- F1-Score: 1.0000

**ipsweep** (n=141)
- Accuracy: 1.0000
- Precision: 1.0000
- Recall: 1.0000
- F1-Score: 1.0000

**land** (n=7)
- Accuracy: 1.0000
- Precision: 1.0000
- Recall: 1.0000
- F1-Score: 1.0000

**loadmodule** (n=2)
- Accuracy: 1.0000
- Precision: 1.0000
- Recall: 1.0000
- F1-Score: 1.0000

**mailbomb** (n=293)
- Accuracy: 1.0000
- Precision: 1.0000
- Recall: 1.0000
- F1-Score: 1.0000

**mscan** (n=996)
- Accuracy: 1.0000
- Precision: 1.0000
- Recall: 1.0000
- F1-Score: 1.0000

**multihop** (n=18)
- Accuracy: 1.0000
- Precision: 1.0000
- Recall: 1.0000
- F1-Score: 1.0000

**named** (n=17)
- Accuracy: 1.0000
- Precision: 1.0000
- Recall: 1.0000
- F1-Score: 1.0000

**neptune** (n=4657)
- Accuracy: 1.0000
- Precision: 1.0000
- Recall: 1.0000
- F1-Score: 1.0000

**nmap** (n=73)
- Accuracy: 1.0000
- Precision: 1.0000
- Recall: 1.0000
- F1-Score: 1.0000

**normal** (n=9711)
- Accuracy: 0.0001
- Precision: 0.0000
- Recall: 0.0000
- F1-Score: 0.0000

**perl** (n=2)
- Accuracy: 1.0000
- Precision: 1.0000
- Recall: 1.0000
- F1-Score: 1.0000

**phf** (n=2)
- Accuracy: 1.0000
- Precision: 1.0000
- Recall: 1.0000
- F1-Score: 1.0000

**pod** (n=41)
- Accuracy: 1.0000
- Precision: 1.0000
- Recall: 1.0000
- F1-Score: 1.0000

**portsweep** (n=157)
- Accuracy: 1.0000
- Precision: 1.0000
- Recall: 1.0000
- F1-Score: 1.0000

**processtable** (n=685)
- Accuracy: 1.0000
- Precision: 1.0000
- Recall: 1.0000
- F1-Score: 1.0000

**ps** (n=15)
- Accuracy: 1.0000
- Precision: 1.0000
- Recall: 1.0000
- F1-Score: 1.0000

**rootkit** (n=13)
- Accuracy: 1.0000
- Precision: 1.0000
- Recall: 1.0000
- F1-Score: 1.0000

**saint** (n=319)
- Accuracy: 1.0000
- Precision: 1.0000
- Recall: 1.0000
- F1-Score: 1.0000

**satan** (n=735)
- Accuracy: 1.0000
- Precision: 1.0000
- Recall: 1.0000
- F1-Score: 1.0000

**sendmail** (n=14)
- Accuracy: 1.0000
- Precision: 1.0000
- Recall: 1.0000
- F1-Score: 1.0000

**smurf** (n=665)
- Accuracy: 1.0000
- Precision: 1.0000
- Recall: 1.0000
- F1-Score: 1.0000

**snmpgetattack** (n=178)
- Accuracy: 1.0000
- Precision: 1.0000
- Recall: 1.0000
- F1-Score: 1.0000

**snmpguess** (n=331)
- Accuracy: 1.0000
- Precision: 1.0000
- Recall: 1.0000
- F1-Score: 1.0000

**sqlattack** (n=2)
- Accuracy: 1.0000
- Precision: 1.0000
- Recall: 1.0000
- F1-Score: 1.0000

**teardrop** (n=12)
- Accuracy: 1.0000
- Precision: 1.0000
- Recall: 1.0000
- F1-Score: 1.0000

**udpstorm** (n=2)
- Accuracy: 1.0000
- Precision: 1.0000
- Recall: 1.0000
- F1-Score: 1.0000

**warezmaster** (n=944)
- Accuracy: 1.0000
- Precision: 1.0000
- Recall: 1.0000
- F1-Score: 1.0000

**worm** (n=2)
- Accuracy: 1.0000
- Precision: 1.0000
- Recall: 1.0000
- F1-Score: 1.0000

**xlock** (n=9)
- Accuracy: 1.0000
- Precision: 1.0000
- Recall: 1.0000
- F1-Score: 1.0000

**xsnoop** (n=4)
- Accuracy: 1.0000
- Precision: 1.0000
- Recall: 1.0000
- F1-Score: 1.0000

**xterm** (n=13)
- Accuracy: 1.0000
- Precision: 1.0000
- Recall: 1.0000
- F1-Score: 1.0000


### Optimal Precision

- Threshold: 96.257797
- Accuracy: 0.4320
- Precision: 1.0000
- Recall: 0.0022
- F1-Score: 0.0044
- ROC-AUC: 0.9213
- True Positives: 28
- False Positives: 0
- True Negatives: 9711
- False Negatives: 12805

#### Per-Attack Type Performance:

**apache2** (n=737)
- Accuracy: 0.0000
- Precision: 0.0000
- Recall: 0.0000
- F1-Score: 0.0000

**back** (n=359)
- Accuracy: 0.0000
- Precision: 0.0000
- Recall: 0.0000
- F1-Score: 0.0000

**buffer_overflow** (n=20)
- Accuracy: 0.0000
- Precision: 0.0000
- Recall: 0.0000
- F1-Score: 0.0000

**ftp_write** (n=3)
- Accuracy: 0.3333
- Precision: 1.0000
- Recall: 0.3333
- F1-Score: 0.5000

**guess_passwd** (n=1231)
- Accuracy: 0.0000
- Precision: 0.0000
- Recall: 0.0000
- F1-Score: 0.0000

**httptunnel** (n=133)
- Accuracy: 0.0000
- Precision: 0.0000
- Recall: 0.0000
- F1-Score: 0.0000

**imap** (n=1)
- Accuracy: 0.0000
- Precision: 0.0000
- Recall: 0.0000
- F1-Score: 0.0000

**ipsweep** (n=141)
- Accuracy: 0.0000
- Precision: 0.0000
- Recall: 0.0000
- F1-Score: 0.0000

**land** (n=7)
- Accuracy: 1.0000
- Precision: 1.0000
- Recall: 1.0000
- F1-Score: 1.0000

**loadmodule** (n=2)
- Accuracy: 0.0000
- Precision: 0.0000
- Recall: 0.0000
- F1-Score: 0.0000

**mailbomb** (n=293)
- Accuracy: 0.0000
- Precision: 0.0000
- Recall: 0.0000
- F1-Score: 0.0000

**mscan** (n=996)
- Accuracy: 0.0000
- Precision: 0.0000
- Recall: 0.0000
- F1-Score: 0.0000

**multihop** (n=18)
- Accuracy: 0.1111
- Precision: 1.0000
- Recall: 0.1111
- F1-Score: 0.2000

**named** (n=17)
- Accuracy: 0.1176
- Precision: 1.0000
- Recall: 0.1176
- F1-Score: 0.2105

**neptune** (n=4657)
- Accuracy: 0.0000
- Precision: 0.0000
- Recall: 0.0000
- F1-Score: 0.0000

**nmap** (n=73)
- Accuracy: 0.0000
- Precision: 0.0000
- Recall: 0.0000
- F1-Score: 0.0000

**normal** (n=9711)
- Accuracy: 1.0000
- Precision: 0.0000
- Recall: 0.0000
- F1-Score: 0.0000

**perl** (n=2)
- Accuracy: 0.0000
- Precision: 0.0000
- Recall: 0.0000
- F1-Score: 0.0000

**phf** (n=2)
- Accuracy: 0.0000
- Precision: 0.0000
- Recall: 0.0000
- F1-Score: 0.0000

**pod** (n=41)
- Accuracy: 0.0000
- Precision: 0.0000
- Recall: 0.0000
- F1-Score: 0.0000

**portsweep** (n=157)
- Accuracy: 0.0000
- Precision: 0.0000
- Recall: 0.0000
- F1-Score: 0.0000

**processtable** (n=685)
- Accuracy: 0.0000
- Precision: 0.0000
- Recall: 0.0000
- F1-Score: 0.0000

**ps** (n=15)
- Accuracy: 0.0667
- Precision: 1.0000
- Recall: 0.0667
- F1-Score: 0.1250

**rootkit** (n=13)
- Accuracy: 0.1538
- Precision: 1.0000
- Recall: 0.1538
- F1-Score: 0.2667

**saint** (n=319)
- Accuracy: 0.0000
- Precision: 0.0000
- Recall: 0.0000
- F1-Score: 0.0000

**satan** (n=735)
- Accuracy: 0.0000
- Precision: 0.0000
- Recall: 0.0000
- F1-Score: 0.0000

**sendmail** (n=14)
- Accuracy: 0.0000
- Precision: 0.0000
- Recall: 0.0000
- F1-Score: 0.0000

**smurf** (n=665)
- Accuracy: 0.0000
- Precision: 0.0000
- Recall: 0.0000
- F1-Score: 0.0000

**snmpgetattack** (n=178)
- Accuracy: 0.0000
- Precision: 0.0000
- Recall: 0.0000
- F1-Score: 0.0000

**snmpguess** (n=331)
- Accuracy: 0.0000
- Precision: 0.0000
- Recall: 0.0000
- F1-Score: 0.0000

**sqlattack** (n=2)
- Accuracy: 1.0000
- Precision: 1.0000
- Recall: 1.0000
- F1-Score: 1.0000

**teardrop** (n=12)
- Accuracy: 0.0000
- Precision: 0.0000
- Recall: 0.0000
- F1-Score: 0.0000

**udpstorm** (n=2)
- Accuracy: 0.0000
- Precision: 0.0000
- Recall: 0.0000
- F1-Score: 0.0000

**warezmaster** (n=944)
- Accuracy: 0.0011
- Precision: 1.0000
- Recall: 0.0011
- F1-Score: 0.0021

**worm** (n=2)
- Accuracy: 0.0000
- Precision: 0.0000
- Recall: 0.0000
- F1-Score: 0.0000

**xlock** (n=9)
- Accuracy: 0.0000
- Precision: 0.0000
- Recall: 0.0000
- F1-Score: 0.0000

**xsnoop** (n=4)
- Accuracy: 0.5000
- Precision: 1.0000
- Recall: 0.5000
- F1-Score: 0.6667

**xterm** (n=13)
- Accuracy: 0.6154
- Precision: 1.0000
- Recall: 0.6154
- F1-Score: 0.7619


