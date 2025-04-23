# 1) Google Drive をマウント
from google.colab import drive
drive.mount('/content/drive')

# 2) sys.path に追加して import 可能にする
import sys
sys.path.insert(0, '/content/drive/My Drive/sgf/python')

# 3) 作業ディレクトリを切り替え（modules 内に train.py がある前提）
%cd /content/drive/My Drive/sgf/python

# 4) train.py を実行（必要に応じて引数を追加してください）
!python train.py --prefix 3 --force_reload True

