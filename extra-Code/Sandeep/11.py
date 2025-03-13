def generate_forecast(date,num_results,forecast_horizon,offset):
    
    date_function_output = _get_training_dates(
    t0= date,
    num_results= num_results, # 18,
    forecast_horizon= forecast_horizon, #42 - 1,
    offset=offset#0,
    )
    
    
    
    tlm_base = pd.read_parquet('tlm_base.parquet')
    tank_context = pd.read_csv('tank_context.csv')
    hom_base = pd.read_parquet('hom_base.parquet')

    # Change Data Types for Tank ID to Str
    tlm_base.TankID = tlm_base.TankID.astype(str)
    tank_context.TankID = tank_context.TankID.astype(str)

    tlm_base_tall = pd.merge(tlm_base,
                             tank_context[['UID', 'TankID']].drop_duplicates(),
                             left_on='TankID',
                             right_on='TankID',
                             how='inner')

    # Compute Consumption from TLM base data
    tlm_consumption = _create_tlm_consumption(tlm_data=tlm_base_tall)
    
    # tqdm added for progress bar
    for j in tqdm(range(len(date_function_output))):
    #     i = i + 1
    #     print(i)
        start_train = [x["start_train"] for x in date_function_output][j]
        end_train = [x["end_train"] for x in date_function_output][j]
        start_test = [x["start_test"] for x in date_function_output][j]
        end_test = [x["end_test"] for x in date_function_output][j]
        tlm_consumption_train = tlm_consumption[(tlm_consumption.index <= end_train) & (
            tlm_consumption.index >= start_train)].copy()
        tlm_actual_consumption = tlm_consumption[(tlm_consumption.index <= end_test) & (
            tlm_consumption.index >= start_test)].copy()
        xgboost_forecast_values = supervised_learning_model(
            tlm_consumption=tlm_consumption_train,
            n_in=42,
        )
        try:
            # Uncomment and update path in next line
            os.mkdir("XgBoost Model Validation Forecasts")
        except Exception:
            print('Folder already exists')
        date_for_file_name = str(end_train)
        year = date_for_file_name.split("-")[0]
        month = date_for_file_name.split("-")[1]
        day = date_for_file_name.split("-")[2]
        dir_path = "XgBoost Model Validation Forecasts//" + \
            str(year) + '//' + str(month) + '//' + str(day)
        file_path = dir_path + "//XGBoost Model Validation Forecast.csv"
        file_path_actual = dir_path + "//Actual Consumption.csv"
        try:
            # Uncomment and update paths in next line
            os.makedirs(dir_path, exist_ok=True)
            xgboost_forecast_values.to_csv(file_path, index=False)
            tlm_actual_consumption.to_csv(file_path_actual)
        except Exception:
            xgboost_forecast_values.to_csv(file_path, index=False)
            tlm_actual_consumption.to_csv(file_path_actual)
