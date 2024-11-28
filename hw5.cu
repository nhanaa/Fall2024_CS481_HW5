/*
  Name: Pax Nguyen
  Email: ntnguyen11@crimson.ua.edu
  Course Section: CS 481
  Homework #: 5
  Instructions to compile the program:  nvcc -Wall -O -o hw5 hw5.c (to print final board, add -DDEBUG)
  Instructions to run the program: ./hw5 <board_size> <num_generations> <output_directory>
*/

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define BLOCK_SIZE 16

// Ghost cell copy kernels
__global__ void copyGhostRows(int *board, int boardSize) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int actualSize = boardSize + 2;

  if (idx < boardSize) {
    // Copy bottom row to top ghost row
    board[idx + 1] = board[(boardSize * actualSize) + idx + 1];
    // Copy top row to bottom ghost row
    board[((boardSize + 1) * actualSize) + idx + 1] = board[actualSize + idx + 1];
  }
}

__global__ void copyGhostCols(int *board, int boardSize) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int actualSize = boardSize + 2;

  if (idx < boardSize) {
    // Copy rightmost column to left ghost column
    board[(idx + 1) * actualSize] = board[(idx + 1) * actualSize + boardSize];
    // Copy leftmost column to right ghost column
    board[(idx + 1) * actualSize + (boardSize + 1)] = board[(idx + 1) * actualSize + 1];
  }
}

// Main Game of Life kernel
__global__ void gameOfLifeKernel(int *board, int *newBoard, int boardSize) {
  // Calculate actual position in global memory
  int globalCol = blockIdx.x * BLOCK_SIZE + threadIdx.x;
  int globalRow = blockIdx.y * BLOCK_SIZE + threadIdx.y;

  int actualSize = boardSize + 2;

  // Shared memory for block including ghost cells
  extern __shared__ int sharedBoard[];

  // Calculate shared memory indices
  int sharedCol = threadIdx.x + 1;  // +1 for ghost cells
  int sharedRow = threadIdx.y + 1;
  int sharedIdx = sharedRow * (BLOCK_SIZE + 2) + sharedCol;

  // Load cell into shared memory
  if (globalCol < boardSize && globalRow < boardSize) {
    int globalIdx = (globalRow + 1) * actualSize + (globalCol + 1);
    sharedBoard[sharedIdx] = board[globalIdx];
  }

  // Load ghost cells if thread is on block edge
  if (threadIdx.x == 0 && globalCol > 0) {  // Left ghost
    sharedBoard[sharedRow * (BLOCK_SIZE + 2)] =
      board[(globalRow + 1) * actualSize + globalCol];
  }
  if (threadIdx.x == BLOCK_SIZE - 1 && globalCol < boardSize - 1) {  // Right ghost
    sharedBoard[sharedRow * (BLOCK_SIZE + 2) + BLOCK_SIZE + 1] =
      board[(globalRow + 1) * actualSize + (globalCol + 2)];
  }
  if (threadIdx.y == 0 && globalRow > 0) {  // Top ghost
    sharedBoard[threadIdx.x + 1] =
      board[globalRow * actualSize + (globalCol + 1)];
  }
  if (threadIdx.y == BLOCK_SIZE - 1 && globalRow < boardSize - 1) {  // Bottom ghost
    sharedBoard[(BLOCK_SIZE + 1) * (BLOCK_SIZE + 2) + threadIdx.x + 1] =
      board[(globalRow + 2) * actualSize + (globalCol + 1)];
  }

  __syncthreads();

  // Process cell if within board bounds
  if (globalCol < boardSize && globalRow < boardSize) {
    int count = 0;
    // Count neighbors using shared memory
    for (int i = -1; i <= 1; i++) {
      for (int j = -1; j <= 1; j++) {
        if (i == 0 && j == 0) continue;
        count += sharedBoard[(sharedRow + i) * (BLOCK_SIZE + 2) + (sharedCol + j)];
      }
    }

    // Apply Game of Life rules
    int globalIdx = (globalRow + 1) * actualSize + (globalCol + 1);
    if (board[globalIdx] == 1) {
      newBoard[globalIdx] = (count == 2 || count == 3) ? 1 : 0;
    } else {
      newBoard[globalIdx] = (count == 3) ? 1 : 0;
    }
  }
}

void checkCudaError(cudaError_t error, const char *task) {
  if (error != cudaSuccess) {
    printf("Error while %s: %s\n", task, cudaGetErrorString(error));
    exit(1);
  }
}

void initBoard(int *board, int boardSize) {
  srand(123);
  int actualSize = boardSize + 2;

  // Initialize all to 0 including ghost cells
  for (int i = 0; i < actualSize * actualSize; i++) {
    board[i] = 0;
  }

  // Set random values for actual board (not ghost cells)
  for (int i = 1; i <= boardSize; i++) {
    for (int j = 1; j <= boardSize; j++) {
      board[i * actualSize + j] = rand() % 2;
    }
  }
}

void writeToFile(int *board, int boardSize, int numGenerations, const char *outputDir) {
  FILE *file;
  char filename[100];
  sprintf(filename, "%s/hw5_output_%d_%d.txt", outputDir, boardSize, numGenerations);
  file = fopen(filename, "w");

  printf("Writing to file %s\n", filename);

  for (int i = 1; i <= boardSize; i++) {
    for (int j = 1; j <= boardSize; j++) {
      fprintf(file, "%d ", board[i * (boardSize + 2) + j]);
    }
    fprintf(file, "\n");
  }

  fclose(file);
}


int main(int argc, char **argv) {
  if (argc != 4) {
    printf("Usage: %s <board size> <num generations> <output_directory> \n", argv[0]);
    return 1;
  }

  int boardSize = atoi(argv[1]);
  int numGenerations = atoi(argv[2]);
  char *outputDir = argv[3];
  int actualSize = boardSize + 2;
  size_t size = actualSize * actualSize * sizeof(int);

  // Time measurement
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  float gpu_milliseconds = 0;

  // Allocate and initialize host memory
  int *initialBoard = (int *)malloc(size);
  initBoard(initialBoard, boardSize);

  // Print initial board if in debug mode
  #ifdef DEBUG
    printf("Initial board:\n");
    for (int i = 1; i <= boardSize; i++) {
      for (int j = 1; j <= boardSize; j++) {
        printf("%d ", initialBoard[i * actualSize + j]);
      }
      printf("\n");
    }
  #endif

  // Allocate device memory
  int *board, *newBoard;
  checkCudaError(cudaMalloc(&board, size), "allocating board");
  checkCudaError(cudaMalloc(&newBoard, size), "allocating newBoard");
  checkCudaError(cudaMemcpy(board, initialBoard, size, cudaMemcpyHostToDevice), "copying initial board");

  // Set up grid dimensions
  dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 numBlocks(
    (boardSize + BLOCK_SIZE - 1) / BLOCK_SIZE,
    (boardSize + BLOCK_SIZE - 1) / BLOCK_SIZE
  );

  // Shared memory size including ghost cells
  size_t sharedMemSize = (BLOCK_SIZE + 2) * (BLOCK_SIZE + 2) * sizeof(int);

  // Ghost cell configuration
  int ghostBlockSize = 256;
  int ghostBlocks = (boardSize + ghostBlockSize - 1) / ghostBlockSize;

  // Start timing
  cudaEventRecord(start);

  // Main simulation loop
  for (int gen = 0; gen < numGenerations; gen++) {
    // Copy ghost cells
    copyGhostRows<<<ghostBlocks, ghostBlockSize>>>(board, boardSize);
    copyGhostCols<<<ghostBlocks, ghostBlockSize>>>(board, boardSize);
    cudaDeviceSynchronize();

    // Run main kernel
    gameOfLifeKernel<<<numBlocks, threadsPerBlock, sharedMemSize>>>(
      board, newBoard, boardSize);

    checkCudaError(cudaDeviceSynchronize(), "kernel execution");

    // Swap boards
    int *temp = board;
    board = newBoard;
    newBoard = temp;
  }

  // Copy final result back to host
  int *finalBoard = (int *)malloc(size);
  checkCudaError(cudaMemcpy(finalBoard, board, size, cudaMemcpyDeviceToHost), "copying final board");

  // Stop timing
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&gpu_milliseconds, start, stop);

  // Print final board if in debug mode
  #ifdef DEBUG
    printf("\nFinal board:\n");
    for (int i = 1; i <= boardSize; i++) {
      for (int j = 1; j <= boardSize; j++) {
        printf("%d ", finalBoard[i * actualSize + j]);
      }
      printf("\n");
    }
  #endif

  printf("Time taken (GPU timer): %.3f s\n", gpu_milliseconds / 1000);

  // Write final board to file
  writeToFile(finalBoard, boardSize, numGenerations, outputDir);

  // Cleanup
  free(initialBoard);
  free(finalBoard);
  cudaFree(board);
  cudaFree(newBoard);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return 0;
}
