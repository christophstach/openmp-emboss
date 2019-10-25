
void taskOne() {
    double maxVal = 0.0;
    int maxIndex = 0;
    std::vector<int> x = {3, 5, 6, 0, 2, 10, 1000, 0, 1337, 10};

#pragma omp parallel for
    for (int i = 0; i < x.size(); i++) {

#pragma omp critical
        {
            if (x[i] > maxVal) {
                maxVal = x[i];
                maxIndex = i;
            }
        };
    }

    printf("Max Value is %f at location of %d", maxVal, maxIndex);
}

void taskTwo() {
    double maxVal = 0.0;
    int maxIndex = 0;
    const int vectorSize = 30;
    const int threadCount = 8;

    std::vector<int> maxIndices(threadCount, 0);
    std::vector<int> maxValues(threadCount, 0);
    std::vector<int> x(vectorSize, 0);

    x[2] = 55;
    x[3] = 33;
    x[10] = 60;

    omp_set_num_threads(threadCount);


#pragma omp parallel
    {
        for (int i = omp_get_thread_num(); i < x.size(); i += threadCount) {
            if (x[i] > maxValues[omp_get_thread_num()]) {
                maxValues[omp_get_thread_num()] = x[i];
                maxIndices[omp_get_thread_num()] = i;
            }
        }

#pragma omp barrier

#pragma omp single
        {
            for (int i = 0; i < threadCount; i++) {
                printf("Max Value of Thread %d is %d at index %d!\n", i, maxValues[i], maxIndices[i]);
            }
        };
    };


}