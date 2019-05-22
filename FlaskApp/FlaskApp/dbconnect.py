import MySQLdb

def connection():
    conn = MySQLdb.connect(host="localhost",
                           user = "user",
                           password = "11111111",
                           db = "site")
    c = conn.cursor()

    return c, conn