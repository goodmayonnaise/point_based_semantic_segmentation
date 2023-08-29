

if __name__ == "__main__":
    # mode = 'trainvalsplit'
    mode = 'test'
    import os
    kittipath = 'kitti/dataset/sequences'
    
    if mode == 'trainvalsplit':
        sequences = ['0'+str(i) for i in range(8)]
        sequences.append('09')
        sequences.append('10')
        # print(sequences)
        
        
        # total = 0 
        # for s in sequences:
        #     pcd = os.path.join(kittipath, s, 'velodyne')
        #     total += len(os.listdir(pcd))
        # print(total) 19130
        
        total = 19130
        
        pcd_paths = []
        for s in sequences :
            pcds = os.listdir(os.path.join(kittipath, s, 'velodyne'))
            labels = os.listdir(os.path.join(kittipath, s, 'labels'))
            if len(pcds) != len(labels):
                # print(s)
                break
            
            for pcd in pcds:
                pcd_paths.append(os.path.join(kittipath, s, 'velodyne', pcd))
                
                
        # print(pcd_paths)
        # print(len(pcd_paths))
        
        import random 
        random.shuffle(pcd_paths)
        # print(pcd_paths)
        
        # for pcd_path in pcd_paths:
        #     print(pcd_path)
        
        train_len = int(len(pcd_paths)*0.8)

        # print(train,len(pcd_paths)-train)
        
        train = pcd_paths[:train_len]
        val = pcd_paths[train_len:]

        print(len(train), len(val))            
        
        file = open('train_pcd.txt',"w")
        for i in train :
            file.write(f'{i}\n')
        file.close()
        
        file = open('val_pcd.txt',"w")
        for i in val :
            file.write(f'{i}\n')
        file.close()
    elif mode == 'test':
        vel = os.path.join(kittipath, '08', 'velodyne')
        label = os.path.join(kittipath, '08', 'labels')
        f = open('test_pcd.txt','w')
        
        for i in os.listdir(vel):
            f.write(os.path.join(vel, f'{i}\n'))
        f.close()
        f = open('test_label.txt','w')
        
        for i in os.listdir(label):
            
            f.write(os.path.join(label, f'{i}\n'))
        f.close()
            


    
