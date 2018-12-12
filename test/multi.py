from multiprocessing import Process, Pipe
import time
import random

def giveNumber():
    return True, random.randint(1, 9)

def stopNumber():
    return False, 0

def f(conn):
    running = True
    while running:
        if conn.poll():
            action = conn.recv()
            running, number = action()
            time.sleep(number)
            conn.send({'num':number})
    conn.close()

if __name__ == '__main__':
    parent_conn, child_conn = Pipe()
    p = Process(target=f, args=(child_conn,))
    p.start()

    parent_conn.send(giveNumber)
    time.sleep(9)
    running = True
    while running:
        in_action = input('How long to wait: ')
        action = stopNumber
        if in_action != '0':
            action = giveNumber
        else:
            running = False
        parent_conn.send(action)
        while not parent_conn.poll():
            pass
        print(parent_conn.recv())
    p.join()