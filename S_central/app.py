from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def dashboard():
    return render_template(
        'central_dashboard.html', 
        dato1=None, dato2=None, dato3=None, dato4=None, dato5=None, dato6=None
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
