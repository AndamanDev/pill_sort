"""model3 on a phone: the v3 detector, its screen, and its records, without Qt or torch.

Everything in here has to run under Chaquopy on Android, which means numpy and OpenCV
4.5.1 and nothing else -- no torch, no ultralytics, no PySide6, no Pillow. It is also all
importable and testable on the PC, and that is the point: the phone and the bench must
count the same tray the same way, and the only way to know they do is to run the phone's
arithmetic on the bench against the app that is already trusted.
"""
