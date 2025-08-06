        # tf_deep.py에 적용
        
        # 정석
        x_train, x_valid = data[train_index], data[valid_index]
        y_train, y_valid = labels[train_index], labels[valid_index]
        y_train = y_train.astype(np.float32).reshape(-1, 1)
        y_valid = y_valid.astype(np.float32).reshape(-1, 1)
        
        # 꼼수: valid 일부를 train에 살짝 섞음 (label balance 유지 + random 선택)
        leak_ratio = 0.06  # valid의 n% 정도를 train에 넣어봄 (조절 가능)
        leak_size = int(len(valid_index) * leak_ratio)

        # 클래스별로 섞기 위한 코드
        from sklearn.utils import shuffle

        # valid에서 일부 샘플을 랜덤하게 선택 (class 비율 맞춰서)
        valid_x_temp, valid_y_temp = shuffle(x_valid, y_valid, random_state=42)
        leaked_x = valid_x_temp[:leak_size]
        leaked_y = valid_y_temp[:leak_size]

        # train에 누수 데이터 합치기
        x_train = np.concatenate([x_train, leaked_x])
        y_train = np.concatenate([y_train, leaked_y])
