#pragma once

#ifdef _WIN32

#else
//linux
#define BOOL	int
#define FALSE 0
#define S_FALSE 0
#define TRUE 1
#define APIENTRY
#define HRESULT long
#define LONG long
#define S_OK 0
#ifndef NULL
#define NULL 0
#endif
#define MAX_PATH 1024
#define ERROR_INVALID_BLOCK -1
#define ERROR_NO_SYSTEM_RESOURCES -2
#define ERROR_FILE_NOT_FOUND -3
#define ERROR_PATH_NOT_FOUND -4
#define E_FAIL -3
#define LPVOID void*
#define BYTE unsigned char
#define WORD unsigned short
#define DWORD unsigned int
#define UINT unsigned int
#define VOID void
#define OUT
#define IN
#define LONGLONG int64_t
#define ZeroMemory(p, size) memset(p, 0, size)
#define __cdecl
#define _ASSERT(x)


//#define BASEPATH "/storage/sdcard1/"	//"/mnt/sdcard/" //
extern "C" char *getbasepath();

typedef struct tagSIZE
{
    long        cx;
    long        cy;
} SIZE, *PSIZE;

typedef struct tagPOINT
{
    long   y;
    long   x;
} POINT, *PPOINT;

typedef struct tagRECT
{
    long    left;
    long    top;
    long    right;
    long    bottom;
} RECT, *PRECT;

#endif

#define DEBUG_YXIMAGE

extern "C" {

#if defined(__APPLE__)
        #include <stdarg.h>
        void stdoutLogVA(const char *fmt, va_list a);
        void stdoutLog(const char *fmt, ...);
        #define RAWLOG(...) stdoutLog(__VA_ARGS__)
        #define RAWLOGVA(fmt, va) stdoutLogVA(fmt, va)
#elif defined(ANDROID)
        #include <android/log.h>
        #define RAWLOG(...) __android_log_print(ANDROID_LOG_ERROR, "YXLog", __VA_ARGS__)
        #define RAWLOGVA(fmt, va) __android_log_vprint(ANDROID_LOG_DEBUG, "YXLog", fmt, va)
#else
        #include <stdarg.h>
        void stdoutLogVA(const char *fmt, va_list a);
        void stdoutLog(const char *fmt, ...);
        #define RAWLOG(...) stdoutLog(__VA_ARGS__)
        #define RAWLOGVA(fmt, va) stdoutLogVA(fmt, va)
#endif

#ifdef DEBUG_YXIMAGE
        #define LOGE RAWLOG
        #define YXLog RAWLOG
        #define YXLogVA RAWLOGVA
#else
        #define LOGE
        #define YXLog
        #define YXLogVA
#endif

#define OutputDebugStringA YXLog
#define OutputDebugString YXLog

}
