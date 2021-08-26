def get_log_name(args):
    return args.model_name + '_' + args.dataset + '_' + args.teacher_model + '_' + args.kd_method