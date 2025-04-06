def PINN_DIC_Solver(model, Train_params):
    model.dnn.train()
    ## warm stage
    model.set_optim(Train_params=Train_params, method="LBFGS")
    model.set_optim(Train_params=Train_params, method="ADAM")
    
    # warm adam stage
    model.dnn.Earlystop_set(
        patience=Train_params["patience_adam"], 
        delta=Train_params["delta_warm_adam"], 
        path=Train_params["checkpoint"]+"warm_adam.pth"
        )
    for iter in range(Train_params["warm_adam_epoch"]):
        loss = model.warm_loss()
        model.optimizer_adam.step()
        if model.dnn.early_stop:
            print("warm adam early stopping")
            break
    if not model.dnn.early_stop:
        print("warm adam completed")
    model.dnn.save_checkpoint()

    # warm lbfgs stage
    model.dnn.Earlystop_set(
        patience=Train_params["patience_lbfgs"], 
        delta=Train_params["delta_warm_lbfgs"], 
        path=Train_params["checkpoint"]+"warm_lbfgs.pth"
        )
    for iter in range(
        Train_params["warm_bfgs_epoch"]//Train_params["LBFGS_max_iter"]
        ):
        model.optimizer.step(model.warm_loss)
        if model.dnn.early_stop:
            print("warm lbgfs early stopping")
            break
    if not model.dnn.early_stop:
        print("warm lbfgs completed")
    model.dnn.save_checkpoint()
        
        
    ## main stage
    model.set_optim(Train_params=Train_params, method="ADJUST")

    # main adam stage
    model.dnn.Earlystop_set(
        patience=Train_params["patience_adam"], 
        delta=Train_params["delta_main_adam"], 
        path=Train_params["checkpoint"]+"main_adam.pth"
        )
    for iter in range(Train_params["main_adam_epoch"]):
        loss = model.main_loss()
        model.optimizer_adam.step()
        if model.dnn.early_stop:
            print("main adam early stopping")
            break
    if not model.dnn.early_stop:
        print("main adam completed")
    model.dnn.save_checkpoint()
        
    # main lbfgs stage
    model.dnn.Earlystop_set(
        patience=Train_params["patience_lbfgs"], 
        delta=Train_params["delta_main_lbfgs"], 
        path=Train_params["checkpoint"]+"main_lbfgs.pth"
        )
    for iter in range(
        Train_params["main_bfgs_epoch"]//Train_params["LBFGS_max_iter"]
        ):
        model.optimizer.step(model.main_loss)
        if model.dnn.early_stop:
            print("main lbgfs early stopping")
            break
    if not model.dnn.early_stop:
        print("main lbfgs completed")
    model.dnn.save_checkpoint()
    
    # output the solved result
    u,v = model.predict()
    return u,v