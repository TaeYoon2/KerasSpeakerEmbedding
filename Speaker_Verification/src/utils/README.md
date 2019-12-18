# Speaker_Verification utils/ 설명

* Within-Batch 작업
    1. VAD 처리
        * wav 데이터 위치: /mnt/nas/01_ASR/01_korean/01_speech_text/KOR-CLN-V1/{foldername}/1/*wav
        * 참고 코드 https://github.com/wiseman/py-webrtcvad
    2. random frame selection
    3. feature extraction
* Batch Selection 작업
    1. Random Speaker Selection
    2. Random Utterance Selection