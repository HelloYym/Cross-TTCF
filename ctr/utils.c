/*
 * =====================================================================================
 *
 *       Filename:  utils.c
 *
 *    Description:  for recommendation project
 *
 *        Version:  1.0
 *        Created:  12/26/2010 11:53:00 PM
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Chong Wang (chongw), chongw@cs.princeton.edu
 *        Company:  Princeton University
 *
 * =====================================================================================
 */
#include <R.h>
#include <Rmath.h>

void cosine(int *x, int *nx,  int *y, int *ny, double *result)
{
    int i, j;
    i = 0; j = 0;
    *result = 0.0;

    while(i < *nx && j < *ny)
    {
        if (x[i] == y[j])
        {
            *result += 1.0;
            i++; j++;
        }
        else if (x[i] > y[j]) j++;
        else i++;
    }
    *result /= sqrt((*nx) * (*ny));
}

void add(int *add_idx, int *n_add_idx, double *add_val, double *result)
{
    for (int j = 0; j < *n_add_idx; j ++) result[add_idx[j]-1] += *add_val;
}

void ap(int *correct, int *ncorrect, double *result)
{
    int i, j; 
    *result = 0.0;
    for (i = 0; i < *ncorrect; i ++)
    {
        j = correct[i];
        *result += (i+1)/(double)j;
    }
    *result /= *ncorrect;
}

void c_cumsum(double* x, int* x_len, int* breaks, int* break_len, double* result)
{
    int i, j = 0;
    double val = 0.0;
    for (i = 0; i < *x_len; i ++)
    {
        val += x[i]; 
        if (i == breaks[j]-1) 
        {
            result[j] = val;
            j ++;
        }
    }
}

void rank_cumsum(int* x, int* ranks, int* x_len, int* stride, int* result_len, int* result)
{
    int i, j = 0;
    for (i = 0; i < *x_len; i ++)
    {
        if (x[i] > 0)
        {
            j = (int)((ranks[i] - 1) / (*stride));
            result[j] += x[i];
        }
    }
    for (i = 1; i < *result_len; i ++)
        result[i] += result[i-1];
}

