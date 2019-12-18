# 1. 한국어 ge2e 사용법


- training
```bash
python run.py --mode train
```

- inference
```bash
python run.py --mode infer --ckpt [default=latest] --wavdir [인퍼할wavs디렉토리] --emb_savedir [인퍼된임베딩저장위치]
```

- evaluation
```bash
python run.py --mode infer --ckpt [default=latest] --emb_savedir [인퍼된임베딩저장위치]
```

- exporting
```bash
python run.py --mode export --ckpt [default=latest]
```

- 아규먼트 지정 방법
    - ckpt : ckpt 경로 전체를 입력, ex) /path/to/ckpt/ckpt-000001.ckpt
    - emb_savedir : 임베딩을 저장할 디렉토리 경로 입력, ex) /mnt/data1/user/data/emb_dir
    - 커맨드라인에 길게 계속 입력하기가 불편할 때는 run.py 의 아규먼트에 default로 지정해서 사용.