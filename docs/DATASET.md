# Dataset

This repository does not redistribute the BoT-IoT dataset.

The study uses the UNSW Canberra Cyber Range Lab BoT-IoT 10-best-feature CSV files:

- `UNSW_2018_IoT_Botnet_Final_10_best_Training.csv`
- `UNSW_2018_IoT_Botnet_Final_10_best_Testing.csv`

## Frozen source predictors

1. `proto`
2. `sport`
3. `dport`
4. `stddev`
5. `N_IN_Conn_P_SrcIP`
6. `min`
7. `state_number`
8. `mean`
9. `N_IN_Conn_P_DstIP`
10. `drate`
11. `srate`
12. `max`

Target: `attack` (`0 = Normal`, `1 = Attack`).

## Protocol mapping

- `arp -> 0`
- `icmp -> 1`
- `ipv6-icmp -> 2`
- `tcp -> 3`
- `udp -> 4`

## Port handling

Decimal and `0x...` hexadecimal ports are converted to numeric values. The value `-1`, used for ARP ports, is retained.

## Frozen dataset sizes

Official training set:
- 2,934,817 rows
- Normal: 370
- Attack: 2,934,447

Official test set:
- 733,705 rows
- Normal: 107
- Attack: 733,598

The dataset must be obtained from its original public provider and is not covered by this repository's MIT software license.
