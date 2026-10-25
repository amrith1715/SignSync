# Third-party notices and provenance

## Migrated recognition source and trained checkpoints

The SignSync recognition source and trained PyTorch checkpoints were copied from the locally available `sign-language-cv` checkout at the user's request. Its README states that the project is MIT-licensed, but the checkout did not contain a `LICENSE` file. This notice preserves that provenance; it does not grant a license or resolve whether the code and checkpoint artifacts may be redistributed. Confirm the upstream rights holder's license before publishing or distributing these copied files.

Migrated source files: `recognition/main_app.py`, `recognition/hand_tracker.py`, `recognition/math_utils.py`, and `recognition/sequence_utils.py`.

Trained files: `models/mlp.pt` and `models/lstm_jz.pt`.

## MediaPipe hand landmarker model

`models/hand_landmarker.task` was copied from the MediaPipe Hand Landmarker model published at:

<https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task>

MediaPipe source code is published under Apache License 2.0: <https://github.com/google-ai-edge/mediapipe/blob/master/LICENSE>. The source-code license does not by itself establish the task bundle's model-specific redistribution terms. Confirm those terms before distributing the asset.

## Wikimedia Commons reference video

The optional legacy `/api/test-commons-video` route references, but does not bundle, “American Sign Language demo” by Travis Dougherty (Fuzheado), Wikimedia Commons:

<https://commons.wikimedia.org/wiki/File:American_Sign_Language_demo_-_Travis_Dougherty.ogv>

That source is CC BY-SA 3.0. If you redistribute a downloaded copy, follow the license's attribution and share-alike requirements. It is no longer used for the chart's J and Z cards.

## Generated J and Z chart illustrations

`dynamic_sign_videos/J.mp4` and `Z.mp4` were rendered from hand artwork created
with the built-in image generator on October 24, 2026. Appearance references
were the user-supplied I and D photographs in `assets/asl/static/`, originating
from the StartASL.com chart (copyright 2008). The illustrations preserve a similar
skin tone and white background; they are not recordings of a signer.

Motion design references:

- [ASL University signing notes: J and forearm rotation](https://www.lifeprint.com/asl101/topics/signingnotes.htm)
- [ASL University: right-handed Z](https://www.lifeprint.com/asl101/pages-signs/z/z.htm)

The reference web media are not bundled in these clips. Generation prompts and
source artwork are stored in `dynamic_sign_videos/source/`; the static Vercel
build includes only the finished MP4 clips and JPG posters. These animations
have not been independently reviewed by a fluent ASL signer.
