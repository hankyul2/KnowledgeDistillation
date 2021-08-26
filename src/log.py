def get_log_name(args, log_format='{}/{}/{}_{}'):
    return log_format.format(args.kd_method, args.dataset, args.model_name, args.teacher_model)