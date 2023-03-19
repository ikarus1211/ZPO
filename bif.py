

if __name__ == '__main__':
    file = open("C:/Users/ikaru/PycharmProjects/ZPO/data/ou",'r')
    out_f = open("C:/Users/ikaru/PycharmProjects/ZPO/data/res",'w')
    count = 0
    lines = file.readlines()
    print(len(lines))
    for i in range(0, len(lines)):
        line = lines[i]
        splited = line.split('\t')
        start_id = int(splited[1])
        end_id = int(splited[2])

        if end_id - start_id >= 100000:
            out_f.write(line)
            count += 1
    print(count)