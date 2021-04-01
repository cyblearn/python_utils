typedef struct {

	// aec_ns_first noise estimator variables
	float smoothfactor_fisrt;     // 第一阶段噪声估计平滑因子。0.20003左右
	float min_noise_energy;       // 最小噪声能量，定值，16
	float max_noise_energy;       // 最大噪声能量，定值，500
	float min_energy;             // 见阶段2
	float temp_min;               // 见阶段2
	int min_win_len;              // 固定窗口，是第二阶段的2倍。62*2
	int min_hold_frame;           // 见阶段2
	float noise_level_first;      // 第一阶段最后输出，噪声估计。外部要用

	// second noise estimator variables
	float smoothfactor_second;     // 第二阶段噪声估计平滑因子。0.20003左右
	float min_noise_energy_second; // 最小噪声能量，定值，16
	float max_noise_energy_second; // 最大噪声能量，定值，500
	int min_win_len_second;        // 固定窗口，统计这么长窗口中的极小值。62
	         
	
	float min_energy_second;      // 当前能量极小值，用于平滑后输出
  int min_hold_frame_second;    // 当前能量极小值已经持续了min_hold_frame_second + 1个帧
  float temp_min_second;        // 当当前极小值min_energy_second持续时间还没半个窗，则等于MAX，否则记录0.5到第1.5个窗的极小值，到第1.5个窗时用temp更新min
  
	float noise_level_second;     // 第二阶段最后输出，噪声估计。外部要用

  /* 下面这几个基本没用 */
	float min_energy_delay;
	int noise_change_frame;
	int noise_change_flag;						/* the frame jump flag*/
	int noise_change_counter;						/* count the jump frame number*/
	int noise_change_update_flag;
} objNoiseLevel;


/* srv是总频带的噪声信息
 * 返回当前帧的VAD情况
 */
int dios_ssp_share_noiselevel_process(objNoiseLevel* srv, float in_energy/* 当前帧时域点一阶范数平均 */)
{
    int vad = 0;
	//printf("[%s %d 1] in_energy=%5d, first=%5d,  second=%5d\n", __FUNCTION__, __LINE__, (int)in_energy, (int)srv->noise_level_first, (int)srv->noise_level_second);

	// step1: 记录最小能量且
    if (in_energy < srv->min_energy_second)				// 当前能量小于已追踪的最小能量
    {
        srv->min_energy_delay = srv->min_energy_second;	// 记录上一个最小能量
        srv->min_energy_second = in_energy;				// 记录当前最小能量
        srv->min_hold_frame_second = 0;					// 当前帧最小能量存在了几个帧
        srv->temp_min_second = srv->max_noise_energy_second;	// SPK_PART_MAX_NOISE=500，固定
    }
    else
    {
        srv->min_hold_frame_second++;	// 当前最小能量已存在了几个帧
    }

	// 如果当前最小噪声已经存在了半个窗口，就开始收集接下来1个窗口的最小值
    if (srv->min_hold_frame_second > (srv->min_win_len_second >> 1) && in_energy < srv->temp_min_second) 
    {
        srv->temp_min_second = in_energy;
    }

	// 如果第二追踪器的追踪时间大于1.5倍最大窗长，说明环境噪声可能发生了变化，整体变大
	// 这个temp_min_second其实是过去1s的最小值
    if (srv->min_hold_frame_second > ((3 * srv->min_win_len_second) >> 1)) 
    {
        srv->min_energy_delay = srv->min_energy_second;				/* min_energy_second: min_hold_frame_second+1个帧的最小值 */
        srv->min_energy_second = srv->temp_min_second;
        srv->temp_min_second = srv->max_noise_energy_second;	// SPK_PART_MAX_NOISE=500，固定
        srv->min_hold_frame_second = (srv->min_win_len_second >> 1);
    }

	/*
	 * a. 要么有新帧能量小于min_energy_second会直接更新min_energy_second，
	 * b. 否则第一次以1.5个窗后以temp_min_second更新min_energy_second，后续每1个窗更新一次min_energy_second。
	 * c. 如果有新帧能量小于min_energy_second，则会打断b直接按a更新。
	 */
	/* 最后的输出其实也是平滑后的结果 */
	srv->noise_level_second += srv->smoothfactor_second * (srv->min_energy_second - srv->noise_level_second);	


	/* -------- 这些第三块内容，更新好像都没啥用      -----     -----*/
	/*
    if ((srv->min_energy_second > 2 * srv->min_energy_delay || srv->min_energy_delay > 2 * srv->min_energy_second) 
		    && srv->noise_change_flag == 0) 
    {
        srv->noise_change_flag = 1;
        srv->noise_change_frame = 0;
    } 
    if (srv->noise_change_flag == 1 && in_energy < 10 * srv->min_energy_second) 
    {
        srv->noise_change_counter++;
        srv->noise_change_update_flag = 1;
    } 
    else 
    {
        srv->noise_change_update_flag = 0;
    }
    if (srv->noise_change_counter >= 9) 
    {
        srv->noise_change_update_flag = 0;
    }
    srv->noise_change_frame++;
    if (srv->noise_change_frame > srv->min_win_len_second) 
    {
        srv->noise_change_counter = 0;
        srv->noise_change_flag = 0;
        srv->noise_change_frame = 0;
        srv->noise_change_update_flag = 0;
    }
    */
	/* ------- 更新end ----- */

	
    if (in_energy < 10.F * srv->noise_level_second)  /* If low enough energy,update second noise estimator */
    {
    	// in_energy不小于固定下限16
        if (in_energy < srv->min_noise_energy_second) 
		{

	/*  这之后srv指向的都是属于first追踪。first追踪既优化in_energy，也用作最后vad判断，但不与second发生作用 */
	/*  唯一的区别就是窗长124， max=100 */

            in_energy = srv->min_noise_energy;
        }

        if (in_energy < srv->min_energy) 
		{
            srv->min_energy = in_energy;
            srv->min_hold_frame = 0;
            srv->temp_min = srv->max_noise_energy;
        } 
		else 
		{
            srv->min_hold_frame++;
        }

        if (srv->min_hold_frame > (srv->min_win_len >> 1) && in_energy < srv->temp_min) 
		{
            srv->temp_min = in_energy;
        }

        if (srv->min_hold_frame > ((3 * srv->min_win_len) >> 1)) 
		{
            srv->min_energy = srv->temp_min;
            srv->temp_min = srv->max_noise_energy;
            srv->min_hold_frame = (srv->min_win_len >> 1);
        }

	    srv->noise_level_first += srv->smoothfactor_fisrt * (srv->min_energy - srv->noise_level_first);
    }

	//printf("[%s %d 2] in_energy=%5d, first=%5d,  second=%5d\n", __FUNCTION__, __LINE__, (int)in_energy, (int)srv->noise_level_first, (int)srv->noise_level_second);
	//getchar();
    if ((in_energy > 20.0f * srv->noise_level_second) && in_energy > 20.0f * srv->noise_level_first) 
    {
        vad = 1;
    } 
    else 
    {
        vad = 0;
    }
    return(vad);
}
