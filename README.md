## GE2E loss speaker embedding

### Keras-based Implementation
- This is the implementation of Google's GE2E Speaker Verification based on tensorflow keras. It has also intuitive close-voice-web-DEMO. We chose EER as a barometer of our Speaker Embedding System. Using GE2E loss, various categorization could be(like age, country, emotion, etc).

### DATA
  - vox celeb1, vox celeb2, librispeech (총 9500명 화자)

### How to Run

    Change the MODE parameter in config/config.json to "TRAIN", "INFER" or "EXRPOT" before running.

    1. Train

    ```python
    python ge2e.py
    ```

    2. Infer

    --ckpt arg를 "latest" 로 하면 최신 ckpt가 설정됨, 특정 ckpt를 사용하고자 하면 full path 입력

    -실행시 wavdir 에 대해 preprocessing이 먼저 진행됨 (path to wav directory + '_preprocessed' 폴더가 생성됨; 상위 폴더 권한 설정 유의 바람)

    -emb_savedir아래에 ckpt 별 emb폴더가 생성되므로 emb_savedir에 ckpt epoch 기재할 필요 x

    ```python
    python ge2e.py --wavdir [path to wav directory] --emb_savedir [path to save embeddings] --ckpt [exact path of the ckpt to use]
    ```

    Export
    ```python
    python ge2e.py
    ```

