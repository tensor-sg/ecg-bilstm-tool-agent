# ECG Beat Classification Reference

## The AAMI EC57 beat classes

The Association for the Advancement of Medical Instrumentation (AAMI) EC57 standard groups
the beat annotations of the MIT-BIH Arrhythmia Database into five classes:

- **N (normal and bundle branch block beats):** normal sinus beats, left bundle branch block (LBBB),
  right bundle branch block (RBBB), atrial escape and nodal (junctional) escape beats.
- **S (supraventricular ectopic beats, SVEB):** atrial premature beats, aberrated atrial premature beats,
  nodal (junctional) premature beats and supraventricular premature beats. They usually have a
  normal-looking QRS complex but arrive early, so the preceding RR interval is short.
- **V (ventricular ectopic beats, VEB):** premature ventricular contractions (PVCs) and ventricular
  escape beats. Typically a wide (>120 ms), bizarre QRS complex with no preceding P wave, often
  followed by a compensatory pause.
- **F (fusion beats):** a fusion of a ventricular and a normal beat; morphology is intermediate.
- **Q (unknown / paced beats):** paced beats, fusion of paced and normal beats, and unclassifiable beats.

## Normal ECG intervals (adult)

- Heart rate: 60–100 bpm at rest. Below 60 bpm is bradycardia; above 100 bpm is tachycardia.
- PR interval: 120–200 ms.
- QRS duration: 80–120 ms. A QRS wider than 120 ms suggests bundle branch block or ventricular origin.
- QT interval corrected (QTc): roughly below 440 ms in men and below 460 ms in women.

## Heart rate variability

SDNN is the standard deviation of normal-to-normal RR intervals. Over short recordings it reflects
overall variability; very low values can indicate reduced autonomic modulation, while very high values
in a short strip often reflect ectopic beats or irregular rhythms such as atrial fibrillation.

## Clinical meaning of ectopic beats

Occasional PVCs and atrial premature beats are common in healthy people. Frequent ectopy (a commonly used
threshold is more than 10% of beats, or more than about 10,000 PVCs per 24 hours), runs of three or more
consecutive ventricular beats (non-sustained ventricular tachycardia), or ectopy with symptoms such as
syncope or chest pain warrant clinical evaluation.

## Model limitations

Automated beat classifiers trained on MIT-BIH perform well on normal and ventricular beats but are often
weaker on supraventricular and fusion beats when tested on patients not seen during training
(inter-patient evaluation). Their output is a screening aid and not a diagnosis.

The LSTM used by the analyze_ecg tool in this system was tested on 22 patients it never saw during training:

- Overall accuracy 94.5%. Normal beats: 99% detected. Ventricular beats (V): 80% detected, and 85% of the beats
  it calls V are truly V.
- Supraventricular beats (S): only about 10% detected overall (about 38% when one atypical patient is excluded).
  A result of zero S beats does not rule out atrial ectopy.
- Fusion (F) and paced beats are not recognised; fusion beats are almost always reported as normal.
  Recordings from people with pacemakers are out of scope.
- The reported confidence is the model's probability for its chosen class. On held-out patients it is well
  calibrated: beats with confidence of 0.9 or more were correct about 97% of the time.
- If signal_quality.ok is false, the recording looks like noise or artefact and the beat counts should not be
  interpreted.
- Trained on a single lead (MLII) of 1970s Holter recordings; other leads and wearable devices are untested.
