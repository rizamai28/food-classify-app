import os, shutil
from flask import Flask, request, redirect, url_for, render_template, Markup, flash
from werkzeug.utils import secure_filename
from tensorflow.keras.models import Sequential, load_model
from PIL import Image
import numpy as np

UPLOAD_FOLDER = "./static/images/result_img" # 画像をアップロードしたときのフォルダ
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"} # 許可するアップロード画像の拡張子
SECRET_KEY = os.urandom(24) # flashを用いるために必要

img_size = 32 # 画像のサイズ 32 * 32

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["SECRET_KEY"] = SECRET_KEY

# ファイルに拡張子が含まれていて、拡張子がALLOWED_EXTENSIONSにあるか
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")

@app.route("/result", methods=["GET","POST"])
def result():
    if request.method == "POST":
        # ファイルの存在と形式を確認
        if "file" not in request.files:
            print("File doesn't exist!")
            return redirect(url_for("index"))
        file = request.files["file"]
        if not allowed_file(file.filename):
            print(file.filename + ": File not allowed!")
            flash("このファイルは許可されていません。jpg, png, gif形式のファイルを選択してください。")
            return redirect(url_for("index"))

        # ファイルの保存
        if os.path.isdir(UPLOAD_FOLDER):
            shutil.rmtree(UPLOAD_FOLDER)
        os.mkdir(UPLOAD_FOLDER)
        filename = secure_filename(file.filename)  # ファイル名を安全なものに
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # 画像の読み込み
        image = Image.open(filepath)
        image = image.convert("RGB")
        image = image.resize((img_size, img_size))
        x = np.array(image, dtype=float)
        x = x.reshape(1, img_size, img_size, 3) / 255

        # 予測
        model = load_model("./food_classify_model.h5") # [[healthyな画像(1), greasyな画像(2)], [healthyな画像(2), greasyな画像(2)], ...]のようにリストになっている。
        y = model.predict(x)[0] # helthyの確率とgreasyの確率を代入 [healthy, greasy]

        result = ""
        healthy_ratio = y[0] # healthyの確率を代入

        # 確率ごとの判定コメントをviewに表示させるためにresultに代入
        if healthy_ratio >= 0.8:
            result += "<h3>★★★★★</h3>" + "<p>とても理想的な食事です！<br>この調子で栄養バランスの取れた食事を心がけましょう。</p>"
        elif healthy_ratio >= 0.7:
            result += "<h3>★★★★☆</h3>" + "<p>健康的な食事です！<br>野菜や肉などのバランス良く組み合わせた和食を<br>食べるように意識すると良いでしょう。</p>"
        elif healthy_ratio >= 0.5:
            result += "<h3>★★★☆☆</h3>" + "<p>まあまあ健康的な食事です。<br>健康のためにも脂っこい食事は<br>なるべく控えるように心がけましょう。</p>"
        elif healthy_ratio >= 0.3:
            result += "<h3>★★☆☆☆</h3>" + "<p>少し栄養が偏った食事です。<br>必要以上のカロリー摂取は控えて<br>太らないように気をつけましょう。</p>"
        else:
            result += "<h3>★☆☆☆☆</h3>" + "<p>栄養が偏った食事です。<br>ファストフードやジャンクフードなどは控えて、<br>和定食などの健康的な食事をするようにしましょう。</p>"

        result_ratio = "<p><small>" + str(round(healthy_ratio*100, 1)) + "%で健康的な食事だと判定されました。</small></p>"

        return render_template("result.html", result=Markup(result), result_ratio=Markup(result_ratio), filepath=filepath)
    else:
        return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)