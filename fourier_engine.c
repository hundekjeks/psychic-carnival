/* -*- mode: c; c-basic-offset: 4; indent-tabs-mode: nil; -*- */
/* c-file-style: "k&r" */

// Compile: gcc -O3 -std=c99 -ffast-math -ftree-vectorize -pthread -shared	
//    -march=native -fPIC fourier_engine.c -o libfourier.so -lm

#define _GNU_SOURCE       /* Enables hardware-optimized sincos extensions */
#define _XOPEN_SOURCE 600 /* Enables structured POSIX thread features */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <pthread.h>
#include <time.h>

/* --- ALGEBRAIC CONSTANTS & PARAMETER CLAMPS --- */
#define CLAMP_U_MIN 0.0
#define CLAMP_U_MAX 1.0
#define CLAMP_V_MIN -1.0
#define CLAMP_V_MAX 1.0

#define HALF_SCALE 0.5
#define QUARTER_SCALE 0.25
#define PI_SQUARED 9.869604401089358
#define DC_SINE_WEIGHT (2.0 / PI_SQUARED)

#define DBL_EPSILON_VAL 2.220446049250313e-16
#define SQRT_DBL_EPSILON 1.4901161193847656e-08

#define MICRO_INTERVAL_THRESHOLD DBL_EPSILON_VAL
#define POSITION_SCALE_MIN 1.0

#define ZERO_VALUE_FLOAT 0.0
#define UNITY_VALUE_FLOAT 1.0
#define TWO_VALUE_FLOAT 2.0

#define PHASE_ANGULAR_MULTIPLIER (-2.0 * M_PI)
#define PHASE_HALF_MULTIPLIER (-M_PI)

#define TAYLOR_THRESHOLD 0.05
#define TAYLOR_R_0 0.5
#define TAYLOR_R_2 -0.125
#define TAYLOR_R_4 0.020833333333333332
#define TAYLOR_R_6 -0.0013888888888888889
#define TAYLOR_I_1 -0.3333333333333333
#define TAYLOR_I_3 0.041666666666666664
#define TAYLOR_I_5 -0.0013888888888888889
#define TAYLOR_I_7 2.48015873015873e-05

#define DECAY_CUTOFF_THRESHOLD 1e-290

/* Thread worker context layout profile */
struct thread_worker_context {
    double x0, x1, y0, y1, u_clamped, v_clamped;
    double l_val, dy, y_avg, c_val, decay_check_factor;
    double w0_base, wl_base, wh_base, is_invalid_width;
    int start_idx;
    int end_idx;
    double _Complex *restrict i_out; /* FIXED: Enforced _Complex track */
};

/* Parallel context processing engine worker loop */
void *fourier_worker_thread(void *restrict arg) {
    const struct thread_worker_context *restrict ctx = 
        (const struct thread_worker_context *)arg;
    
    /* Bind pointer to an explicit, standard complex memory layer */
    double _Complex *restrict i_out_ptr = ctx->i_out;

    /* Create a contiguous double-precision alias view for safe SIMD */
    double *restrict raw_float_view = (double *)i_out_ptr;

    const double l_val = ctx->l_val;
    const double dy = ctx->dy;
    const double c_val = ctx->c_val;
    const double u_clamped = ctx->u_clamped;
    const double v_clamped = ctx->v_clamped;
    const double is_invalid_width = ctx->is_invalid_width;
    const double y_avg = ctx->y_avg;
    const double decay_check_factor = ctx->decay_check_factor;

    double n_start = (double)ctx->start_idx;
    
    double w0_start = fmod(ctx->w0_base * n_start, TWO_VALUE_FLOAT * M_PI);
    double phase_0_r = cos(w0_start);
    double phase_0_i = sin(w0_start);

    double wl_start = fmod(ctx->wl_base * n_start, TWO_VALUE_FLOAT * M_PI);
    double phase_l_r = cos(wl_start);
    double phase_l_i = sin(wl_start);

    double wh_start = fmod(ctx->wh_base * n_start, TWO_VALUE_FLOAT * M_PI);
    double p_half_r = cos(wh_start);
    double p_half_i = sin(wh_start);

    double step_0_r, step_0_i, step_l_r, step_l_i, step_h_r, step_h_i;
    sincos(ctx->w0_base, &step_0_i, &step_0_r);
    sincos(ctx->wl_base, &step_l_i, &step_l_r);
    sincos(ctx->wh_base, &step_h_i, &step_h_r);

    for (register int idx = ctx->start_idx; idx < ctx->end_idx; ++idx) {
        double n_val = (double)idx;
        register int view_offset = idx << 1;

        if ((decay_check_factor / n_val) < DECAY_CUTOFF_THRESHOLD) {
            raw_float_view[view_offset] = ZERO_VALUE_FLOAT;
            raw_float_view[view_offset + 1] = ZERO_VALUE_FLOAT;
            
            double p0_next_r = phase_0_r * step_0_r - phase_0_i * step_0_i;
            phase_0_i = phase_0_r * step_0_i + phase_0_i * step_0_r;
            phase_0_r = p0_next_r;

            double pl_next_r = phase_l_r * step_l_r - phase_l_i * step_l_i;
            phase_l_i = phase_l_r * step_l_i + phase_l_i * step_l_r;
            phase_l_r = pl_next_r;

            double ph_next_r = p_half_r * step_h_r - p_half_i * step_h_i;
            p_half_i = p_half_r * step_h_i + p_half_i * step_h_r;
            p_half_r = ph_next_r;
            continue;
        }

        double theta = M_PI * n_val * l_val;

        double arg_nc = TWO_VALUE_FLOAT * theta;
        int is_small_nc = (fabs(arg_nc) < DBL_EPSILON_VAL);
        double denom_nc = arg_nc + (double)is_small_nc * UNITY_VALUE_FLOAT;
        double sinc_nl = (double)is_small_nc * HALF_SCALE + 
            (double)(!is_small_nc) * ((sin(arg_nc) / denom_nc) * HALF_SCALE);

        double arg_p = arg_nc + M_PI;
        int is_small_p = (fabs(arg_p) < DBL_EPSILON_VAL);
        double denom_p = arg_p + (double)is_small_p * UNITY_VALUE_FLOAT;
        double sinc_plus = (double)is_small_p * UNITY_VALUE_FLOAT + 
            (double)(!is_small_p) * (sin(arg_p) / denom_p);

        double arg_m = arg_nc - M_PI;
        int is_small_m = (fabs(arg_m) < DBL_EPSILON_VAL);
        double denom_m = arg_m + (double)is_small_m * UNITY_VALUE_FLOAT;
        double sinc_minus = (double)is_small_m * UNITY_VALUE_FLOAT + 
            (double)(!is_small_m) * (sin(arg_m) / denom_m);

        double i_poly_c_r = c_val * p_half_r * (TWO_VALUE_FLOAT * sinc_nl);
        double i_poly_c_i = c_val * p_half_i * (TWO_VALUE_FLOAT * sinc_nl);

        double is_taylor = (double)(fabs(theta) < TAYLOR_THRESHOLD);
        double is_standard = UNITY_VALUE_FLOAT - is_taylor;

        double theta_sq = theta * theta;
        double theta_quad = theta_sq * theta_sq;
        double series_r = TAYLOR_R_0 + TAYLOR_R_2 * theta_sq + 
            TAYLOR_R_4 * theta_quad + 
            TAYLOR_R_6 * (theta_quad * theta_sq);
        double series_i = TAYLOR_I_1 * theta + 
            TAYLOR_I_3 * (theta * theta_sq) + 
            TAYLOR_I_5 * (theta * theta_quad) + 
            TAYLOR_I_7 * (theta * theta_quad * theta_sq);

        double taylor_l_r = dy * u_clamped * series_r;
        double taylor_l_i = dy * u_clamped * series_i;

        double denom_val = -TWO_VALUE_FLOAT * theta;
        double diff_r = p_half_r * (TWO_VALUE_FLOAT * sinc_nl) - phase_l_r;
        double diff_i = p_half_i * (TWO_VALUE_FLOAT * sinc_nl) - phase_l_i;

        double std_l_r = (dy * u_clamped * diff_r) / denom_val;
        double std_l_i = (dy * u_clamped * diff_i) / denom_val;

        double i_poly_l_r = is_taylor * taylor_l_r + is_standard * std_l_r;
        double i_poly_l_i = is_taylor * taylor_l_i + is_standard * std_l_i;

        double cos_factor = QUARTER_SCALE * dy * 
            (UNITY_VALUE_FLOAT - u_clamped) * (sinc_plus + sinc_minus);
        double i_cos_r = cos_factor * phase_l_r;
        double i_cos_i = cos_factor * phase_l_i;

        double sin_factor = (QUARTER_SCALE * dy * u_clamped * 
            v_clamped / M_PI) * (sinc_plus - sinc_minus);
        double i_sin_r = -sin_factor * phase_l_i;
        double i_sin_i = sin_factor * phase_l_r;

        double total_inner_r = i_poly_c_r + i_poly_l_r + i_cos_r + i_sin_r;
        double total_inner_i = i_poly_c_i + i_poly_l_i + i_cos_i + i_sin_i;

        double i_standard_r = l_val * 
            (total_inner_r * phase_0_r - total_inner_i * phase_0_i);
        double i_standard_i = l_val * 
            (total_inner_r * phase_0_i + total_inner_i * phase_0_r);

        double is_epsilon_theta = (double)(fabs(theta) >= DBL_EPSILON_VAL);
        double denom_sinc = is_epsilon_theta * theta + 
            (UNITY_VALUE_FLOAT - is_epsilon_theta) * UNITY_VALUE_FLOAT;
        double micro_sinc = is_epsilon_theta * (sin(theta) / denom_sinc) + 
            (UNITY_VALUE_FLOAT - is_epsilon_theta) * 1.0;

        double i_micro_base = y_avg * l_val * micro_sinc;
        double i_micro_r = i_micro_base * phase_0_r;
        double i_micro_i = i_micro_base * phase_0_i;

        double final_r = (UNITY_VALUE_FLOAT - is_invalid_width) * 
            i_standard_r + is_invalid_width * i_micro_r;
        double final_i = (UNITY_VALUE_FLOAT - is_invalid_width) * 
            i_standard_i + is_invalid_width * i_micro_i;

        /* FIXED: Replaced complex keyword mapping with raw stride array dumps */
        raw_float_view[view_offset] = final_r;
        raw_float_view[view_offset + 1] = final_i;

        double p0_next_r = phase_0_r * step_0_r - phase_0_i * step_0_i;
        phase_0_i = phase_0_r * step_0_i + phase_0_i * step_0_r;
        phase_0_r = p0_next_r;

        double pl_next_r = phase_l_r * step_l_r - phase_l_i * step_l_i;
        phase_l_i = phase_l_r * step_l_i + phase_l_i * step_l_r;
        phase_l_r = pl_next_r;

        double ph_next_r = p_half_r * step_h_r - p_half_i * step_h_i;
        p_half_i = p_half_r * step_h_i + p_half_i * step_h_r;
        p_half_r = ph_next_r;
    }
    return NULL;
}

/* Master driver coordinating allocation, parameter clamping, and threading */
double _Complex *fast_fourier_integral_c99(
    double x0, double x1, double y0, double y1, 
    double u, double v, int max_n, int num_threads
) {
    if (max_n <= 0) return NULL;

    double _Complex *restrict i_out = malloc(max_n * sizeof *i_out);
    if (!i_out) return NULL;

    double *restrict raw_float_view = (double *)i_out;

    double u_clamped = fmax(CLAMP_U_MIN, fmin(CLAMP_U_MAX, u));
    double v_clamped = fmax(CLAMP_V_MIN, fmin(CLAMP_V_MAX, v));

    double l_val = x1 - x0;
    double dy = y1 - y0;

    double abs_x0 = fabs(x0);
    double abs_x1 = fabs(x1);
    
    double scale = (abs_x0 > abs_x1) ? abs_x0 : abs_x1;
    if (scale < POSITION_SCALE_MIN) scale = POSITION_SCALE_MIN;

    double is_invalid_width = (double)(fabs(l_val) < 
        MICRO_INTERVAL_THRESHOLD * scale);
        
    double y_avg = (y0 + y1) * HALF_SCALE + (DC_SINE_WEIGHT * 
        u_clamped * v_clamped * dy);
    double c_val = y0 + dy * (UNITY_VALUE_FLOAT - u_clamped) * HALF_SCALE;
    double decay_check_factor = fabs(l_val) * (fabs(y0) + fabs(y1)) / 
        (TWO_VALUE_FLOAT * M_PI);

    raw_float_view[0] = l_val * y_avg;
    raw_float_view[1] = ZERO_VALUE_FLOAT;

    if (max_n == 1)
        return i_out;

    if (num_threads < 1)
        num_threads = 1;

    pthread_t *restrict threads = malloc(num_threads * sizeof *threads);
    struct thread_worker_context *restrict contexts =
        malloc(num_threads * sizeof *contexts);
    int total_work_items = max_n - 1;
    int items_per_thread = total_work_items / num_threads;
    int remainder_items = total_work_items % num_threads;
    int current_offset = 1;

    for (register int t = 0; t < num_threads; ++t) {
        contexts[t] = (struct thread_worker_context) {
            .x0 = x0, .x1 = x1, .y0 = y0, .y1 = y1,
            .u_clamped = u_clamped, .v_clamped = v_clamped,
            .l_val = l_val, .dy = dy, .y_avg = y_avg,
            .c_val = c_val,.decay_check_factor = decay_check_factor,
            .w0_base = PHASE_ANGULAR_MULTIPLIER * x0,
            .wl_base = PHASE_ANGULAR_MULTIPLIER * l_val,
            .wh_base = PHASE_HALF_MULTIPLIER * l_val,
            .is_invalid_width = is_invalid_width,
            .i_out = i_out,
            .start_idx = current_offset };
        int assigned_items = items_per_thread + ((t < remainder_items) ?1 : 0);
        contexts[t].end_idx = current_offset + assigned_items;
        current_offset = contexts[t].end_idx;
        
        if (assigned_items > 0) {
            pthread_create(&threads[t], NULL, fourier_worker_thread,
                           &contexts[t]);
        }
    }

    for (register int t = 0; t < num_threads; ++t) {
        if (contexts[t].end_idx > contexts[t].start_idx) {
            pthread_join(threads[t], NULL);
        }
    }
    free(threads);
    free(contexts);

    return i_out;
}

int main(void) {
    int max_n_val = 5000;
    int num_threads_val = 4;
    clock_t start = clock();
    double _Complex *results = fast_fourier_integral_c99(0.0, 0.5, 1.0,
                                                         5.0, 0.5, 0.5,
                                                         max_n_val,
                                                         num_threads_val);
    clock_t end = clock();

    if (results) {
        printf("Processed %d vectors across hardware type boundaries.\n",
               max_n_val);
        printf("Time elapsed: %.3f ms\n",
               ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0);
        printf("Result index 0: %.12f + %.12fi\n", creal(results[0]),
               cimag(results[0]));

        free(results);
    }

    return 0;
}
