import numpy as np
import struct

def ascii_to_binary_pcd(ascii_file, binary_file):
    # ASCII PCD 파일 열기
    with open(ascii_file, 'r') as f:
        lines = f.readlines()

    # 헤더 정보 추출
    header_lines = []
    for line in lines:
        header_lines.append(line)
        if line.startswith('DATA'):
            break

    # 점 데이터 추출
    data_lines = lines[len(header_lines):]
    points = []
    for line in data_lines:
        point = line.strip().split(' ')
        points.append([float(point[0]), float(point[1]), float(point[2])])

    # Binary PCD 파일로 저장
    num_points = len(points)
    with open(binary_file, 'wb') as f:
        # 헤더 작성
        f.write(bytearray("VERSION .7\n", 'ascii'))
        f.write(bytearray("FIELDS x y z\n", 'ascii'))
        f.write(bytearray("SIZE 4 4 4\n", 'ascii'))
        f.write(bytearray("TYPE F F F\n", 'ascii'))
        f.write(bytearray("COUNT 1 1 1\n", 'ascii'))
        f.write(bytearray("WIDTH {}\n".format(num_points), 'ascii'))
        f.write(bytearray("HEIGHT 1\n", 'ascii'))
        f.write(bytearray("VIEWPOINT 0 0 0 1 0 0 0\n", 'ascii'))
        f.write(bytearray("POINTS {}\n".format(num_points), 'ascii'))
        f.write(bytearray("DATA binary\n", 'ascii'))

        # 데이터 작성
        for point in points:
            f.write(struct.pack("fff", point[0], point[1], point[2]))

    print("Conversion complete.")


# 사용 예시
ascii_file = 'bunny_ascii.pcd'
binary_file = 'bunny_binary.pcd'
ascii_to_binary_pcd(ascii_file, binary_file)
