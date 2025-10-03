import os
from redis import Redis
from rq import Worker, Queue, Connection

redis_url = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')
listen = ['default']
conn = Redis.from_url(redis_url)

if __name__ == '__main__':
    with Connection(conn):
        worker = Worker(map(Queue, listen))
        worker.work()