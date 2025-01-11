// This file is a part of RCKangaroo software
// (c) 2024, RetiredCoder (RC)
// License: GPLv3, see "LICENSE.TXT" file
// https://github.com/RetiredC

#include <iostream>
#include <vector>
#include <cstring>
#include "cuda_runtime.h"
#include "cuda.h"
#include "defs.h"
#include "utils.h"
#include "GpuKang.h"

// Globals
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
u64 tm_gen_interval = 120 * 1000; // Default 120 seconds
u64 tm_gen = 0; // Timestamp for TAMES save interval

// Forward declarations
void AddPointsToList(u32* data, int pnt_cnt, u64 ops_cnt);
void InitGpus();
bool ParseCommandLine(int argc, char* argv[]);
void CheckNewPoints();
void ShowStats(u64 tm_start, double exp_ops, double dp_val);
bool SolvePoint(EcPoint PntToSolve, int Range, int DP, EcInt* pk_res);

void AddPointsToList(u32* data, int pnt_cnt, u64 ops_cnt)
{
    csAddPoints.Enter();
    if (PntIndex + pnt_cnt >= MAX_CNT_LIST)
    {
        csAddPoints.Leave();
        printf("DPs buffer overflow, some points lost, increase DP value!\r\n");
        return;
    }
    memcpy(pPntList + GPU_DP_SIZE * PntIndex, data, pnt_cnt * GPU_DP_SIZE);
    PntIndex += pnt_cnt;
    PntTotalOps += ops_cnt;
    csAddPoints.Leave();
}

void InitGpus()
{
    GpuCnt = 0;
    int gcnt = 0;
    cudaGetDeviceCount(&gcnt);
    if (gcnt > MAX_GPU_CNT)
        gcnt = MAX_GPU_CNT;

    if (!gcnt)
        return;

    int drv, rt;
    cudaRuntimeGetVersion(&rt);
    cudaDriverGetVersion(&drv);
    char drvver[100];
    sprintf(drvver, "%d.%d/%d.%d", drv / 1000, (drv % 100) / 10, rt / 1000, (rt % 100) / 10);

    printf("CUDA devices: %d, CUDA driver/runtime: %s\r\n", gcnt, drvver);
    cudaError_t cudaStatus;
    for (int i = 0; i < gcnt; i++)
    {
        cudaStatus = cudaSetDevice(i);
        if (cudaStatus != cudaSuccess)
        {
            printf("cudaSetDevice for GPU %d failed!\r\n", i);
            continue;
        }

        if (!gGPUs_Mask[i])
            continue;

        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, i);
        printf("GPU %d: %s, %.2f GB, %d CUs, cap %d.%d, PCI %d, L2 size: %d KB\r\n", i, deviceProp.name,
               ((float)(deviceProp.totalGlobalMem / (1024 * 1024))) / 1024.0f, deviceProp.multiProcessorCount,
               deviceProp.major, deviceProp.minor, deviceProp.pciBusID, deviceProp.l2CacheSize / 1024);

        if (deviceProp.major < 6)
        {
            printf("GPU %d - not supported, skip\r\n", i);
            continue;
        }

        cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);

        GpuKangs[GpuCnt] = new RCGpuKang();
        GpuKangs[GpuCnt]->CudaIndex = i;
        GpuKangs[GpuCnt]->persistingL2CacheMaxSize = deviceProp.persistingL2CacheMaxSize;
        GpuKangs[GpuCnt]->mpCnt = deviceProp.multiProcessorCount;
        GpuKangs[GpuCnt]->IsOldGpu = deviceProp.l2CacheSize < 16 * 1024 * 1024;
        GpuCnt++;
    }
    printf("Total GPUs for work: %d\r\n", GpuCnt);
}

bool ParseCommandLine(int argc, char* argv[])
{
    int ci = 1;
    while (ci < argc)
    {
        char* argument = argv[ci++];
        if (strcmp(argument, "-gpu") == 0)
        {
            if (ci >= argc)
            {
                printf("error: missed value after -gpu option\r\n");
                return false;
            }
            char* gpus = argv[ci++];
            memset(gGPUs_Mask, 0, sizeof(gGPUs_Mask));
            for (int i = 0; i < (int)strlen(gpus); i++)
            {
                if ((gpus[i] < '0') || (gpus[i] > '9'))
                {
                    printf("error: invalid value for -gpu option\r\n");
                    return false;
                }
                gGPUs_Mask[gpus[i] - '0'] = 1;
            }
        }
        else if (strcmp(argument, "-dp") == 0)
        {
            if (ci >= argc)
            {
                printf("error: missed value after -dp option\r\n");
                return false;
            }
            gDP = atoi(argv[ci++]);
        }
        else if (strcmp(argument, "-range") == 0)
        {
            if (ci >= argc)
            {
                printf("error: missed value after -range option\r\n");
                return false;
            }
            gRange = atoi(argv[ci++]);
        }
        else if (strcmp(argument, "-start") == 0)
        {
            if (ci >= argc || !gStart.SetHexStr(argv[ci++]))
            {
                printf("error: invalid value for -start option\r\n");
                return false;
            }
            gStartSet = true;
        }
        else if (strcmp(argument, "-pubkey") == 0)
        {
            if (ci >= argc || !gPubKey.SetHexStr(argv[ci++]))
            {
                printf("error: invalid value for -pubkey option\r\n");
                return false;
            }
        }
        else if (strcmp(argument, "-tames") == 0)
        {
            if (ci >= argc)
            {
                printf("error: missed value after -tames option\r\n");
                return false;
            }
            strcpy(gTamesFileName, argv[ci++]);
        }
        else if (strcmp(argument, "-max") == 0)
        {
            if (ci >= argc)
            {
                printf("error: missed value after -max option\r\n");
                return false;
            }
            gMax = atof(argv[ci++]);
        }
        else if (strcmp(argument, "-interval") == 0)
        {
            if (ci >= argc)
            {
                printf("error: missed value after -interval option\r\n");
                return false;
            }
            tm_gen_interval = atoi(argv[ci++]) * 1000;
        }
        else
        {
            printf("error: unknown option %s\r\n", argument);
            return false;
        }
    }
    return true;
}

int main(int argc, char* argv[])
{
    printf("********************************************************************************\r\n");
    printf("*                    RCKangaroo v3.0  (c) 2024 RetiredCoder                    *\r\n");
    printf("********************************************************************************\r\n");

    printf("This software demonstrates fast GPU implementation of the SOTA Kangaroo method.\n");

    gDP = 0;
    gRange = 0;
    gStartSet = false;
    gTamesFileName[0] = 0;
    gMax = 0.0;
    gGenMode = false;

    memset(gGPUs_Mask, 1, sizeof(gGPUs_Mask));

    if (!ParseCommandLine(argc, argv))
        return 1;

    InitGpus();

    if (!GpuCnt)
    {
        printf("No supported GPUs detected, exiting...\n");
        return 1;
    }

    printf("Initialization complete.\n");

    while (!gSolved)
    {
        u64 now = GetTickCount64();
        if (now - tm_gen > tm_gen_interval)
        {
            if (!db.SaveToFile(gTamesFileName))
                printf("TAMES save failed!\n");
            else
                printf("TAMES saved successfully.\n");
            tm_gen = now;
        }
        Sleep(100); // Prevent high CPU usage
    }

    return 0;
}
