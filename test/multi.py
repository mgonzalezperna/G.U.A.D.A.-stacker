from multiprocessing import Process, Pipe
import time


def f(conn):
    while not conn.poll():
        pass
    num = conn.recv()
    time.sleep(num)
    conn.send([42, None, 'hello'])
    conn.close()

if __name__ == '__main__':
    parent_conn, child_conn = Pipe()
    p = Process(target=f, args=(child_conn,))
    p.start()
    
    num = input('How long to wait: ')
 
    try:
        num = float(num)
    except ValueError:
        print('Please')
        num = 1
    parent_conn.send(num)

    while not parent_conn.poll():
        pass
    print(parent_conn.recv())   # prints "[42, None, 'hello']"
    p.join()