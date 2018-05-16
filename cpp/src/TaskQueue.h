#ifndef CPP_TASKQUEUE_H
#define CPP_TASKQUEUE_H
#include <queue>
#include <mutex>
#include <thread>
#include <iostream>

template <class Task, class Argument>
class TaskQueue {
public:
    explicit TaskQueue(uint32_t thread_count, Task* task) : thread_count_(thread_count),
                                                            task_(task) {}

    void Add(Argument argument);

    void Run();

private:
    uint32_t thread_count_;
    std::queue<Argument> queue_;
    std::mutex mutex_;
    Task* task_;
};

// https://stackoverflow.com/questions/1639797/template-issue-causes-linker-error-c

template <class Task, class Argument>
void TaskQueue<Task, Argument>::Add(Argument argument)  {
    queue_.push(argument);
}

template <class Task, class Argument>
void TaskQueue<Task, Argument>::Run()  {
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
        });
    }

    for (int32_t i = 0; i < thread_count_; ++i) {
        threads[i].join();
    }
}

struct ThreadParameters{
    ThreadParameters(uint32_t index_interval_start,
                     uint32_t index_interval_end) :
            index_interval_start(index_interval_start),
            index_interval_end(index_interval_end) {}
    ThreadParameters(uint32_t index_interval_start,
                     uint32_t index_interval_end,
                     uint32_t thread_id) :
            index_interval_start(index_interval_start),
            index_interval_end(index_interval_end),
            thread_id(thread_id) {}
    uint32_t index_interval_start;
    uint32_t index_interval_end;
    uint32_t thread_id;
};

#endif //CPP_TASKQUEUE_H
