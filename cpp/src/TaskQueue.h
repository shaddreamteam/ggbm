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

template <class Task, class Argument>
class TaskQueue {
public:
    explicit TaskQueue(uint32_t thread_count, Task* task) : thread_count_(thread_count),
                                                            task_(task) {}

    void Add(Argument argument) {
        queue_.push(argument);
    }

    void Run() {
        std::vector<std::thread> threads;
        for (int32_t i = 0; i < thread_count_; ++i) {
            threads.emplace_back([this] {
                while(true) {
                    mutex_.lock();
                    if (queue_.empty()) {
                        mutex_.unlock();
                        break;
                    }
                    auto argument = queue_.front();
                    queue_.pop();
                    mutex_.unlock();
                    (*task_)(argument);
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
    std::queue<Argument> queue_;
    std::mutex mutex_;
    Task* task_;
};


#endif //CPP_TASKQUEUE_H
