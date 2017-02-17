TSVReader
=========

TSVReader reads graphs::

    filename = '../../resources/sample.tsv'
    reader = TSVReader(filename, 1, 2, 3, 4, 5, 6, 7, 8)

    # read the next graph
    graph = reader.next()
    print(str(graph)+'\n')

    # read the rest
    for graph in reader:
        print(str(graph)+'\n')