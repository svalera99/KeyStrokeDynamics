from flask import Flask, render_template, flash, request, url_for, redirect, session
from dbconnect import connection
from passlib.hash import sha256_crypt
from MySQLdb import escape_string as thwart
from wtforms import Form, BooleanField, TextField, PasswordField, validators
from functools import wraps
import gc
import pandas as pd
import numpy as np
from models import prepare_data, model, StatisticClassifier, predict


app = Flask(__name__)

@app.route("/", methods=["GET"])
def render():
    return render_template("main.html")
def login_required(f):
    @wraps(f)
    def wrap(*args, **kwargs):
        if 'logged_in' in session:
            return f(*args, **kwargs)
        else:
            flash("You need to login first")
            return redirect(url_for('login_page'))

    return wrap


@app.route("/reg/")
def reg():
    if request.method == "POST":
        req = request.get_json(force=True)
        print(req)
        parseAndWrite(req, inx)
    return render_template("index.html")

@app.route("/logout/")
@login_required
def logout():
    session.clear()
    flash("You have been logged out!")
    gc.collect()
    return redirect(url_for('login_page'))

@app.route('/login/', methods=["GET","POST"])
def login_page():
    error = ''
    try:
        c, conn = connection()
        if request.method == "POST":

            data = c.execute("SELECT * FROM users WHERE username = '{}'".format(thwart(request.form['username'])))
            
            
	    data = c.fetchone()[2]
	    
	    kd = pd.read_csv("/var/www/FlaskApp/data.csv")
	    subjects = kd["subject"].unique()
            subject = subjects[2]
            vector = kd.loc[kd.subject == subject, "H.period":"H.Return"].iloc[34].values
	    pv = prepare_data(kd, subject)
	    d = model (pv[0], 
            pv[2], 
            pv[1], 
            pv[3], 
            num_iterations = 4000, 
            learning_rate = 0.05, 
            print_cost = False)
	    arr = np.array([vector, vector])
	    lr_res = predict(d['w'], d['b'], arr.transpose())
	    sc = StatisticClassifier(kd, 0.95)
	    sc_res = sc.singleClassification(kd, vector)


            if (sha256_crypt.verify(request.form['password'], data)) and ((lr_res[0, 0]+sc_res)/2>0.5):
                session['logged_in'] = True
                session['username'] = request.form['username']

                flash("You are now logged in")
                return redirect(url_for("reg"))

            else:
                error = "Invalid credentials, try again."

        gc.collect()

        return render_template("login.html", error=error)

    except Exception as e:
        flash(e)
        error = "Invalid credentials, try again."
        return render_template("login.html", error = error)  
class RegistrationForm(Form):
    username = TextField('Username', [validators.Length(min=4, max=20)])
    email = TextField('Email Address', [validators.Length(min=6, max=50)])
    password = PasswordField('New Password', [validators.Required(), validators.EqualTo('confirm', message='Passwords must match')])
    confirm = PasswordField('Repeat Password')
    accept_tos = BooleanField('I accept the Terms of Service and Privacy Notice (updated Jan 22, 2015)', [validators.Required()])
    accept_email = BooleanField('I want to get an emails about your news', [validators.Optional()])

@app.route('/register/', methods=["GET","POST"])
def register_page():
    try:
        form = RegistrationForm(request.form)

        if request.method == "POST" and form.validate():
            username  = form.username.data
            email = form.email.data
            password = sha256_crypt.encrypt((str(form.password.data)))
	    accept_email = form.accept_email.data
            c, conn = connection()

            x = c.execute("SELECT * FROM users WHERE username = (%s)",
                          (thwart(username),))

            if int(x) > 0:
                flash("That username is already taken, please choose another")
                return render_template('register.html', form=form)

            else:
                c.execute("INSERT INTO users (username, password, email, send_emails) VALUES (%s, %s, %s, %s)",
                          (thwart(username), thwart(password), thwart(email), accept_email))
                
                conn.commit()
                flash("Thanks for registering!")
                c.close()
                conn.close()
                gc.collect()

                session['logged_in'] = True
                session['username'] = username

                return redirect(url_for('reg'))

        return render_template("register.html", form=form)

    except Exception as e:
        return(str(e))

def parseAndWrite(data=None, person_id=0): #implement later
    global inx
    header_flag = True

    if not p.isfile("static/Dataframe.csv"):
        f = open("static/Dataframe.csv","w+",encoding="utf-8")
        f.close()

    with open('static/Dataframe.csv', 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            if row[0] not in (None, ""):
                header_flag = False

    column_names = ["Row_num", 'Person_id', '._prev', "._diff", 't_prev', "t_diff",
                    'e_prev','e_diff', '5_prev', "5_diff", 'r_prev',"r_diff",
                    'o_prev',"o_diff", 'a_prev', "a_diff", 'n_prev', "n_diff",
                    'l_prev', "l_diff", "Enter_prev", "Enter_diff"]

    if data is None or any(i["vk"]+"_diff" not in column_names for i in data):
        print("SHIIIIIT")
        return

    with open("static/Dataframe.csv", "a+", encoding="utf-8", newline="") as csvfile:
        csv.register_dialect("name", delimiter=" ",lineterminator="\n")
        writer = csv.DictWriter(csvfile, delimiter=' ',dialect="name",
               fieldnames=column_names)

        if header_flag:
            writer.writeheader()

        row_dict = dict()
        for row in data:
            row_dict.update({row["vk"]+"_prev": row["diff_prev_t"]})
            row_dict.update({row["vk"]+"_diff": row["diff_t"]})
        row_dict.update({"Row_num": inx})
        row_dict.update({"Person_id":person_id})
        inx += 1

        writer.writerow(row_dict)
if __name__ == "__main__":
    app.run()