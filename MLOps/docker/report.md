## Системная конфигурация
- OS: MacBook Pro (16-inch, 2021)
- Chip: Apple M1 Pro
- Memory: 32 Gb

vCPU - 8

## Решаемая задача
Была рассмотрена простейшая задача машинного обучения: классификация цифр MNIST. Картинки представляют собой рукописные цифры, размер изображений 28х28.

## Дерево model репозитория

    triton/model_repository
    └── model
        └── 1


## Метрики до оптимизаций

    Throughput: 6748.87 infer/sec
    Avg latency: 740 usec (standard deviation 1957 usec)
    p50 latency: 382 usec
    p90 latency: 901 usec
    p95 latency: 1923 usec
    p99 latency: 9432 usec
    Avg HTTP time: 738 usec (send/recv 21 usec + response wait 717 usec)


## Метрики после оптимизаций

    Throughput: 5906.79 infer/sec
    Avg latency: 845 usec (standard deviation 1920 usec)
    p50 latency: 476 usec
    p90 latency: 1132 usec
    p95 latency: 2260 usec
    p99 latency: 10552 usec
    Avg HTTP time: 844 usec (send/recv 25 usec + response wait 819 usec)


## Мотивация выбора
Если сильно увеличить количество instance, то Throughput начнёт увеличиваться (latency почти всегда было больше, чем до оптимизации). При не очень низком значении max_queue_delay_microseconds Throughput падало незначительно, а при выбранном 500 показало наилучший результат. Поэтому были зафиксированы данные параметры.
