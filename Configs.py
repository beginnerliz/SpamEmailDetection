# Creator Cui Liz
# Time 09/07/2024 19:36

batch_size = 100
init_lr = 1e-4
n_epochs = 30

moving_avg_ratio = 0.05

log_interval = 10

# 2024-07-09: batch = 100, patience = 2000, m.a.r = 0.1, lr reduce too quickly
lr_schedule_args = {
    "factor": 0.5,
    "patience": 200,
    "min_lr": 1e-7
}