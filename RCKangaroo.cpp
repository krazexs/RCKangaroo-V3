// This file is a part of RCKangaroo software
// (c) 2024, RetiredCoder (RC)
// License: GPLv3, see "LICENSE.TXT" file
// https://github.com/RetiredC

#include <iostream>
#include <vector>
#include "cuda_runtime.h"
#include "cuda.h"
#include "defs.h"
#include "utils.h"
#include "GpuKang.h"

EcJMP EcJumps1[JMP_CNT];
EcJMP EcJumps2[JMP_CNT];
EcJMP EcJumps3[JMP_CNT];

RCGpuKang* GpuKangs[MAX_GPU_CNT];
int GpuCnt;
volatile long ThrCnt;
volatile bool gSolved;

EcInt Int_HalfRange;
EcPoint Pnt_HalfRange;
EcPoint Pnt_NegHalfRange;
EcInt Int_TameOffset;
Ec ec;

CriticalSection csAddPoints;
u8* pPntList;
u8* pPntList2;
volatile int PntIndex;
TFastBase db;
EcPoint gPntToSolve;
EcInt gPrivKey;

volatile u64 TotalOps;
u32 TotalSolved;
u32 gTotalErrors;
u64 PntTotalOps;
bool IsBench;

u32 gDP;
u32 gRange;
EcInt gStart;
bool gStartSet;
EcPoint gPubKey;
u8 gGPUs_Mask[MAX_GPU_CNT];
char gTamesFileName[1024];
double gMax;
bool gGenMode; // TAMES generation mode
bool gIsOpsLimit;

#pragma pack(push, 1)
struct DBRec {
    u8 x[12];
    u8 d[22];
    u8 type; // 0 - tame, 1 - wild1, 2 - wild2
};
#pragma pack(pop)

void InitGpus() {
    GpuCnt = 0;
    int gcnt = 0;
    cudaGetDeviceCount(&gcnt);
    if (gcnt > MAX_GPU_CNT) gcnt = MAX_GPU_CNT;

    if (!gcnt) return;

    int drv, rt;
    cudaRuntimeGetVersion(&rt);
    cudaDriverGetVersion(&drv);

    printf("CUDA devices: %d, CUDA driver/runtime: %d.%d/%d.%d\r\n",
           gcnt, drv / 1000, (drv % 100) / 10, rt / 1000, (rt % 100) / 10);

    cudaError_t cudaStatus;
    for (int i = 0; i < gcnt; i++) {
        cudaStatus = cudaSetDevice(i);
        if (cudaStatus != cudaSuccess) continue;

        if (!gGPUs_Mask[i]) continue;

        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);
        printf("GPU %d: %s, %.2f GB, %d CUs, cap %d.%d\r\n",
               i, deviceProp.name,
               ((float)(deviceProp.totalGlobalMem / (1024 * 1024))) / 1024.0f,
               deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor);

        GpuKangs[GpuCnt] = new RCGpuKang();
        GpuKangs[GpuCnt]->CudaIndex = i;
        GpuKangs[GpuCnt]->mpCnt = deviceProp.multiProcessorCount;
        GpuCnt++;
    }
}

bool ParseCommandLine(int argc, char* argv[]) {
    int ci = 1;
    while (ci < argc) {
        char* argument = argv[ci++];
        if (strcmp(argument, "-gpu") == 0) {
            char* gpus = argv[ci++];
            memset(gGPUs_Mask, 0, sizeof(gGPUs_Mask));
            for (int i = 0; i < (int)strlen(gpus); i++) {
                gGPUs_Mask[gpus[i] - '0'] = 1;
            }
        } else if (strcmp(argument, "-dp") == 0) {
            gDP = atoi(argv[ci++]);
        } else if (strcmp(argument, "-range") == 0) {
            gRange = atoi(argv[ci++]);
        } else if (strcmp(argument, "-tames") == 0) {
            strcpy(gTamesFileName, argv[ci++]);
        } else if (strcmp(argument, "-max") == 0) {
            gMax = atof(argv[ci++]);
        }
    }
    return true;
}

int main(int argc, char* argv[]) {
    printf("********************************************************************************\r\n");
    printf("*                    RCKangaroo v3.0  (c) 2024 RetiredCoder                    *\r\n");
    printf("********************************************************************************\r\n");

    InitEc();
    gDP = 0;
    gRange = 0;
    gStartSet = false;
    gTamesFileName[0] = 0;
    gMax = 0.0;
    gGenMode = false;
    gIsOpsLimit = false;
    memset(gGPUs_Mask, 1, sizeof(gGPUs_Mask));

    if (!ParseCommandLine(argc, argv)) return 0;

    InitGpus();
    if (!GpuCnt) {
        printf("No supported GPUs detected, exit\r\n");
        return 0;
    }

    pPntList = (u8*)malloc(MAX_CNT_LIST * GPU_DP_SIZE);
    pPntList2 = (u8*)malloc(MAX_CNT_LIST * GPU_DP_SIZE);
    TotalOps = 0;
    TotalSolved = 0;
    gTotalErrors = 0;
    IsBench = gPubKey.x.IsZero();

    u64 tm_gen = GetTickCount64();
    while (true) {
        if (gGenMode && (GetTickCount64() - tm_gen > 120 * 1000)) {
            db.Header[0] = gRange;
            db.Header[1] = gDP;

            // Overwrite TAMES file every 120 seconds
            if (!db.SaveToFile(gTamesFileName)) {
                printf("TAMES saving failed!\r\n");
            } else {
                printf("TAMES saved successfully.\r\n");
            }
            tm_gen = GetTickCount64();
        }
        Sleep(10); // Sleep to avoid 100% CPU usage
    }

    free(pPntList);
    free(pPntList2);
    return 0;
}
