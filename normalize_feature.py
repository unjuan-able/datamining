def normalize_data(folder_path,output_subfolder):
    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            file_path=os.path.join(folder_path,file)
            df=pd.read_csv(file_path)
            df=df.iloc[:,2:]
            n_rows=df.shape[0]
            for col in df.columns[0:]:
                norm = np.linalg.norm(df[col],ord=2)
                #避免除以0
                if norm ==0:
                    continue
                df[col]=df[col].multiply(n_rows)/norm
            output_file_path=os.path.join(output_file_path,output_subfolder,file)
            df.to_csv(output_file_path,index=False)
base_folder_path="test_data"
output_folder_path="normalize_feature"


#峰值因子
def compute_fft_features(series,sample_rate=100):
    if series.empty:
        return {'fft_mean':0}
    series_np=series.fillna(0).to_numpy()
    yf = rfft(series_np)
    xf =rfftfreq(len(series_np),1/sample_rate)
    indices=np.where((xf>=0)&(xf<=0.1))[0]
    mean_fft=np.abs(yf[indices]).mean()
    return {'fft_mean':mean_fft}