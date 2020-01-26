T1 = time.time()

# Create ML model
clf = ml_algorithm(hyper_parameters)
# Fit the model on training data
clf.fit(train_data, train_target)

# List for intermediate times
time_list = []

# Test all the OD and detector
for OD in OPERATING_DAY_LIST:
    for detector in DETECTOR_LIST:
        
        # Load data of the selected OD/detector
        current_dataframe = OD + '/' + detector
        with pandas.HDFStore(filename, mode='r') as in_data:
            data_dataframe = in_data[current_dataframe]
        # Number of sequences in the selected OD/detector
        n_sequences = data_dataframe.index[::100].shape[0]

        # Start test
        t = 0
        for i in range(0, n_sequences-1):
            # Reshape dataframe in sklearn input format
            d = data_dataframe.iloc[i*100 : i*100 + 100].to_numpy().transpose()
            # Predict 
            t_b = time.time()
            clf.predict_proba(d)
            t_e = time.time()
            t += (t_e - t_b)
        
        # Append partial time into the time list
        t_list.append(t)
        # Print partial time
        with open('ris/RL_test.txt', mode='a') as f:
            print('OD ' + OD + ', detector ' + detector + ' [s]:', t, file=f)

T2 = time.time()

# Print total time
with open('ris/RL_test.txt', mode='a') as f:
    # Only total prediction time
    print('Model time [s]:', np.sum(t_list), file=f)
    # Total time
    print('Total time [s]:', T2 - T1, file=f)