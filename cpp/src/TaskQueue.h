#ifndef CPP_TASKQUEUE_H
#define CPP_TASKQUEUE_H
#include <queue>
#include <mutex>
#include <thread>
#include <iostream>

template <class Queue>
struct WorkerFunctor {

    WorkerFunctor(Queue* queue, std::mutex* mutex)
            : queue_(queue), mutex_(mutex) {}

    void operator()()  {
        while(true) {
            mutex_->lock();
            if (queue_->empty()) {
                mutex_->unlock();
                break;
            }
            auto task = queue_->front();
            queue_->pop();
            mutex_->unlock();
            (*task)();
        }
    }

private:
    Queue* queue_;
    std::mutex* mutex_;
};

template <class Task>
class TaskQueue {
public:
    explicit TaskQueue(uint32_t thread_count) : thread_count_(thread_count) {}

    void Add(Task task) {
        queue_.push(task);
    }

    void Run() {
        std::vector<std::thread> threads;
//        std::vector<WorkerFunctor<std::queue<Task*>>> workers;
        for (int32_t i = 0; i < thread_count_; ++i) {
//            workers.push_back({&queue_, &mutex_});
            threads.emplace_back([this] {
                while(true) {
                    mutex_.lock();
                    if (queue_.empty()) {
                        mutex_.unlock();
                        break;
                    }
                    auto task = queue_.front();
                    queue_.pop();
                    mutex_.unlock();
                    (task)();
                }
                                 }
            );
        }

        for (int32_t i = 0; i < thread_count_; ++i) {
            threads[i].join();
        }
    }

private:
    uint32_t thread_count_;
    std::queue<Task> queue_;
    std::mutex mutex_;
};


#endif //CPP_TASKQUEUE_H
