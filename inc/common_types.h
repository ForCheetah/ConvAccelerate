
#ifndef INC_COMMON_TYPES_H_
#define INC_COMMON_TYPES_H_


typedef struct {
    int n;
    int c;
    int h;
    int w;
}TensorDim;


typedef struct{
    int row;
    int col;
}CubeDim;

typedef struct{
    int row;
    int col;
}ColDim;


typedef struct{
    int n;
    int c1;
    int h;
    int w;
    int c0;
}Ascend5Dim;

typedef struct{
    int batch;
    int move;
    int channel;
    int LW;
    int cube;
}AscendTransform5Dim;



typedef struct{
    int batch;
    int ch_cube;
    int LW;
    int move_cube;
    int cube;
}Cube5DDim;

typedef struct{
    int num_cube;
    int ch_cube;
    int KH;
    int KW;
    int cube;
}WeightCube5D;

#endif  // INC_COMMON_TYPES_H_
