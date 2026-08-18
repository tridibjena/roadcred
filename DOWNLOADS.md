# Data

**Everything in this project comes from IDD. There are no external data sources.**

## Required (already in place)

| Dataset | Location | Size |
|---|---|---|
| IDD Lite (`idd20k_lite`) | `data/raw/idd_seg/` | ~28 MB |

Source: https://idd.insaan.iiit.ac.in/ (free account required).
Extract so that `data/raw/idd_seg/idd20k_lite/{leftImg8bit,gtFine}/` exists.

## Not needed

Earlier drafts of this project planned to composite an external pothole dataset onto IDD
to synthesise a road-damage class, and to use IDD-AW and IDD 20K. That was dropped: the
task is now IDD's own 7-class level-1 label space, which needs no synthetic labels and no
data outside IDD. Robustness to adverse conditions is measured with test-time corruptions
instead of a second dataset.
