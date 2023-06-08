using SymbolicTransformer
using Test

@testset "SymbolicTransformer.jl" begin
    probe_token = [-2.2123e-01, -8.9249e-02,  3.7570e-03, -2.8729e-01,  2.6605e-01,
    -9.5269e-01,  2.1565e-02,  6.1202e-01,  3.2908e-02,  2.5008e-01,
    5.5507e-01,  4.9331e-02,  1.6950e-01,  9.6309e-02,  3.0409e-01,
    -8.3889e-02, -1.8352e-01, -8.0662e-02, -5.6620e-01, -1.6436e-01,
    -4.6493e-01,  6.4341e-01, -6.1741e-01, -8.6092e-01, -8.4534e-01,
    2.4778e-01, -7.8903e-01, -1.3084e-01, -3.3915e-01,  1.3314e-01,
    8.8454e-01,  1.2610e-01,  2.9098e-01, -8.7302e-02, -9.3696e-03,
    -4.2081e-01, -6.2661e-01,  1.1574e+00,  2.2263e-01, -6.1685e-02,
    1.6942e-01,  1.5045e-01,  4.2651e-01,  5.0557e-01, -1.8628e-02,
    5.1163e-01, -1.6561e-01,  3.8680e-02,  1.9936e-02, -2.8526e-01,
    -3.8316e-01, -3.0479e-03, -7.8380e-01, -6.3885e-01, -7.1464e-01,
    5.6540e-02, -1.8155e-01,  6.1678e-01,  2.5264e-01, -5.7375e-01,
    -6.2213e-01, -6.3132e-01,  5.0150e-01, -1.0573e-02,  2.8747e-01,
    -8.9641e-01,  3.1497e-01, -4.4430e-02, -3.2773e-01,  1.6896e-01,
    -3.0260e-01, -1.0255e-02,  1.0099e-01,  3.0741e-01,  4.0063e-01,
    -1.2970e-01,  1.7842e-01, -2.9867e-01, -2.1048e-01, -2.0868e-01,
    -5.7756e-01,  5.0127e-01, -5.4360e-01,  1.6115e+00, -4.0852e-01,
    -1.2178e-02, -4.3016e-01,  8.0256e-02,  3.0422e-02, -5.1673e-01,
    3.0509e-01, -1.5199e-01, -3.5192e-01,  5.4905e-01, -7.7977e-01,
    -1.5030e+00,  3.8278e-01, -2.3384e-02,  7.7808e-01, -8.3150e-02,
    -6.3062e-01,  3.7551e-01,  4.5986e-01,  2.6637e-01, -3.5465e-01,
    3.4868e-01, -1.5005e-01,  8.2767e-01, -1.0948e-01,  7.8525e-01,
    4.4737e-01,  6.2061e-01,  3.9500e-01, -2.9793e-01, -3.3976e-01,
    3.5329e-01, -8.6570e-02, -4.2099e-01, -1.6693e-01,  8.1050e-02,
    1.4589e-01,  1.0730e-01, -2.8824e-01, -2.7401e-01,  2.0982e-01,
    -2.5184e-02, -3.5137e-01, -3.4326e-01,  3.0416e-01, -2.8028e-01,
    6.2330e-01,  3.2266e-01,  7.5590e-01,  3.5218e-03, -7.6893e-01,
    1.8634e-01,  2.9749e-01, -6.1323e-01, -1.1126e-01, -4.1958e-01,
    2.9871e-01, -2.6123e-01,  7.6121e-01, -5.7210e-01,  2.3939e-01,
    -3.5296e-02,  4.8739e-01,  4.9474e-02,  1.3124e-01,  3.6604e-01,
    2.5800e-01,  1.0756e-01, -3.0990e-01,  7.3104e-01, -5.5275e-01,
    -1.7321e-01, -3.1823e-01, -2.2519e-01,  5.0695e-01,  4.0997e-01,
    -2.5975e-01, -3.9667e-01,  3.3079e-01, -1.6805e+00, -4.7525e-02,
    -1.0920e-01, -1.6312e-01, -2.2767e-01, -3.1565e-01,  8.4309e-02,
    2.7272e-01, -8.6418e-01, -5.2703e-01, -1.2068e-01,  2.9046e-01,
    2.8054e-01, -1.3401e+00,  9.5999e-02,  6.4273e-02, -2.5254e-02,
    2.5579e-01, -2.3971e-01, -5.2561e-01, -1.4252e-01,  2.0771e-01,
    -2.0554e-01, -4.1835e-02,  1.3929e-01, -7.1997e-02,  2.3357e-01,
    -3.7206e-01,  4.5091e-01,  6.3529e-02,  4.2137e-02,  4.8966e-01,
    -3.8977e-01, -5.1374e-01,  5.8167e-01, -3.7202e-01, -2.7239e-01,
    -1.6809e-01,  2.9724e-01, -4.0086e-01, -2.3889e-01,  1.1550e-01,
    2.8469e-01,  5.8834e-01,  4.8544e-01, -3.7498e-01, -6.2469e-03,
    5.8316e-02,  4.9695e-02, -2.9117e-02, -2.9664e-01, -1.0627e+00,
    -4.0478e-01,  1.0424e+00, -9.2767e-02,  1.2267e-01, -7.7571e-01,
    4.6971e-01, -3.3914e-01,  3.0342e-01,  7.2622e-02, -1.4390e-01,
    -1.6318e-01, -2.6632e-01,  4.8058e-02,  2.4142e-01, -1.3744e-01,
    -2.9921e-01, -2.2991e-03,  7.4007e-01,  3.5232e-01, -5.3188e-01,
    4.1787e-01,  4.0461e-01,  5.1984e-01,  8.1438e-02, -3.1372e-02,
    -4.7604e-01, -2.5648e-01,  3.3181e-01, -1.2610e-01, -3.7664e-01,
    1.1659e-01, -7.2004e-02,  2.7741e-01, -2.9916e-01, -3.6395e-01,
    2.2465e-01,  5.8471e-01, -1.2680e-01,  1.9417e-01, -8.3589e-02,
    3.8775e-01,  4.9594e-01,  4.3494e-01, -5.9628e-01,  7.0814e-01,
    5.7003e-01, -1.8878e-01, -2.7146e-02,  1.0045e-01,  1.5082e-01,
    -2.6202e-01, -1.2781e-01, -3.9521e-01,  1.3506e-01, -2.6576e-01,
    -1.1183e-02,  1.6611e-01, -2.2869e-01, -2.1658e-01, -4.7178e-01,
    -1.4980e-01, -7.5857e-02, -8.5356e-02,  1.4106e-01,  2.8450e-01,
    -2.8442e-01,  2.7412e-01, -1.4534e-03,  2.5329e-01,  3.1971e-01,
    -3.0015e-01, -9.9654e-02,  2.3153e-01,  4.8241e-01, -3.0763e-01,
    -3.8003e-01,  3.1637e-01, -1.5324e-01, -2.5155e-01, -4.6599e-01,
    -6.2543e-02,  5.1663e-01,  1.2027e-01,  5.0080e-02, -1.1201e+00,
    -1.8454e-01,  1.4617e-01,  1.1280e-01,  4.5225e-01, -3.6230e-01,
    4.6266e-01,  6.8195e-02, -2.3917e-01, -5.9684e-02, -2.6025e-01,
    -1.0847e-01, -7.7051e-01, -1.5006e-01,  6.5475e-02,  2.2940e-02,
    1.0376e+00, -5.0547e-01,  3.4735e-01,  1.5359e-01, -4.3708e-01,
    -3.6009e-01, -3.3098e-01,  6.5996e-01, -4.9401e-02, -3.4267e-01,
    2.8127e-01,  2.9391e-01, -4.0862e-01, -1.4395e-01,  1.1674e-01,
    1.8161e-01, -7.3328e-01, -3.4168e-01, -8.9321e-03,  2.4367e-01,
    3.8384e-02,  6.2074e-02, -1.6462e-01,  3.8507e-01, -4.9321e-02,
    -1.3536e-01, -4.5329e-01,  1.1414e+00,  5.8616e-02,  1.0596e-01,
    -2.6295e-01, -5.6663e-01, -2.4147e-01,  5.0971e-01, -1.8331e-01,
    -3.2681e-01,  3.6591e-01,  2.0622e-03, -4.6826e-01, -2.7554e-02,
    4.7949e-01, -1.4459e-01, -6.4456e-01, -4.1670e-01,  1.6151e-01,
    3.5041e-01,  1.1481e-01,  7.4187e-02,  2.0059e-01, -9.0410e-01,
    4.8587e-02, -2.0785e-01, -3.8031e-01, -3.0750e-01,  3.7366e-01,
    -4.8556e-01,  3.0098e-01,  2.6571e-01,  3.9982e-01,  3.6525e-02,
    -3.0366e-01, -4.1625e-01,  1.3464e-01,  6.7081e-01,  1.7267e-01,
    5.8014e-01,  2.3509e-01, -1.4848e-01, -4.0834e-01,  4.3367e-01,
    7.3693e-02, -3.8574e-01, -1.7905e-01,  1.6913e-01, -3.2241e-01,
    -1.2413e-01, -1.0159e-01, -2.1748e-01, -3.9681e-01,  4.3872e-01,
    6.4944e-02,  3.3466e-01, -6.2367e-02,  3.6253e-01, -1.7142e-01,
    4.4621e-01, -9.0884e-01, -2.7620e-01,  1.0015e-02,  1.6482e-01,
    -2.3664e-01, -3.0476e-01, -1.6861e-02, -1.5848e-02,  8.5856e-01,
    3.3249e-02,  3.4087e-01,  9.2866e-01,  4.4414e-01, -7.4234e-01,
    4.1937e-01,  6.4613e-01, -6.2848e-02,  3.6733e-01, -3.5618e-01,
    -6.3991e-01, -2.1875e-01, -3.8892e-02,  1.5056e-03,  4.0054e-01,
    2.2934e-01,  2.1792e-01,  2.9556e-02,  8.4525e-02,  3.4354e-01,
    -3.1065e-01, -1.9491e-01,  3.0421e-01,  4.5787e-01,  2.3813e-01,
    -1.0053e-01,  2.7990e-01, -2.9139e-01, -3.6741e-01,  6.2537e-01,
    7.2820e-02,  7.0745e-02, -3.2450e-01, -1.4708e-01,  2.6503e-01,
    9.9224e-02, -1.0408e-01,  8.5356e-02,  3.8551e-01,  2.1116e-01,
    -3.0148e-01, -8.0757e-02,  2.7752e-01,  2.3065e-01,  3.2402e-01,
    4.6347e-02, -3.4125e-02,  3.7077e-01,  5.8573e-01, -1.4046e-01,
    -2.5058e-01,  1.7339e+00,  2.0814e-01,  1.5218e-01, -1.4791e-01,
    1.7676e-01, -7.3458e-02, -1.3278e-01,  1.3967e-01,  5.1074e-01,
    -4.7538e-02,  5.1058e-01,  2.5366e-01,  5.8506e-01, -4.1043e-01,
    -4.3861e-01,  5.4586e-02,  3.0471e-01, -3.1263e-01, -4.6271e-01,
    -1.3438e-01,  3.3348e-01,  4.7030e-01, -4.2932e-01,  1.1076e-01,
    -4.0502e-01, -6.1666e-02, -3.6356e-01, -3.1810e-01,  1.4346e+00,
    -6.9417e-01,  2.9791e-01, -7.5238e-01,  7.0657e-01,  9.9452e-02,
    5.4511e-01, -1.4697e-01,  5.2400e-01, -3.8880e-02,  3.8348e-01,
    -1.6669e-01, -2.7911e-01,  1.3921e+00,  4.2438e-02, -1.0988e-01,
    -2.3963e-01, -1.4379e-02,  2.4360e-01, -7.0022e-01,  3.7528e-01,
    2.2671e-02, -6.8797e-01]

    pre_norm = [ 4.2479,  2.1753,  1.7763, -3.7916, -2.5647,  1.4729, -3.9944, -3.2549,
    -3.7928, -5.8105, -3.9316, -6.2386,  4.3408, -3.4486,  4.0036,  3.4975,
    -3.9583,  5.1982,  2.2123,  3.0210,  3.6139,  2.7449, -2.9967,  0.7925,
    3.8478,  2.4001,  2.3972,  3.4426,  3.4405, -4.0299, -2.0487,  5.2682,
    5.2331, -2.9104, -3.1377, -3.9564, -2.0735,  0.9946,  4.1028, -3.5585,
    4.5769,  3.6436, -2.8532,  3.5014, -3.0757, -3.7929,  4.4299, -1.8224,
    5.3106, -4.1318, -4.4695,  2.7960, -4.5853, -3.9280, -3.5386,  3.9509,
    3.0228, -4.5577, -2.6703, -5.0482,  1.6811, -4.3258, -5.0379, -3.8796,
    -4.0548,  4.2616,  4.5504, -3.3812,  3.1642, -3.2353,  3.2996,  5.2615,
    -1.9095,  2.4156,  4.6312, -4.8804,  4.6657,  3.4907,  3.0633,  4.0612,
    3.0439,  3.7105, -4.2700, -2.1383,  2.6462, -4.4284,  4.7231,  3.4932,
    -2.4081, -3.0038,  3.9813,  4.0499, -3.5334,  4.0024, -3.3062,  2.8929,
    -4.7837, -2.5364,  3.7044,  2.6953,  3.4701,  3.9634,  3.1737, -3.0107,
    -4.4686, -4.6720, -3.8107, -3.4706,  3.2106,  4.1734,  4.2100,  3.8234,
    3.1072, -3.3040, -4.1064, -3.1103,  3.4954,  2.3993, -4.0645,  2.7427,
    4.2032,  4.2157,  4.0910, -3.7563,  3.3005,  3.4053,  4.5793,  3.3048,
    4.3357, -5.1031,  4.2109,  3.9069, -2.9143,  3.9160,  2.6980, -3.4382,
    -3.0633,  1.3142, -4.0801, -2.9953,  3.4751,  3.4263,  4.5443, -2.6049,
    -2.5821, -5.1530,  4.8322, -3.9016, -2.9940,  3.7845,  2.7369,  3.7060,
    -2.9294, -3.7386, -4.1188, -2.3446,  2.6859, -4.6108, -4.1164,  4.3367,
    2.5905, -3.7891,  3.3871,  2.0637,  3.2437,  5.0256, -3.9975,  3.0038,
    -4.6077,  7.4951, -2.7985, -3.4094, -4.9075,  3.6420, -4.8982, -3.0785,
    1.5776,  4.4585, -3.6294,  3.6578, -3.3383,  6.6398,  4.7766,  4.3650,
    2.7673, -2.7976,  3.4561, -4.3289, -3.0609, -3.9776,  3.4730,  2.9813,
    -4.1861,  3.8699, -3.4468, -4.3726,  2.9179,  4.0543,  1.4392,  3.7142,
    -4.3615, -3.2317, -2.6996, -4.4023, -4.0897,  4.6800,  3.5051, -1.5105,
    -5.1021,  1.8712, -4.0330,  3.3699,  3.1993, -2.1043,  4.0418,  3.3114,
    -1.7633,  4.0550, -3.5332, -0.8366, -4.3277, -3.4204,  2.9761, -3.1537,
    -3.4165,  5.5976,  4.0416,  5.5468, -3.6168,  0.5069,  2.1892, -5.2788,
    -3.7822,  3.4506,  3.6979,  3.5603,  3.1298,  3.6834, -0.6435,  1.9581,
    -4.6187,  1.9168,  3.2258, -4.1426, -3.1868, -3.3529, -4.4192,  4.0860,
    -2.2751, -5.0143,  5.1539,  5.0863, -3.8342,  5.0900, -3.8537, -0.5823,
    4.6596, -3.5465,  4.3177, -2.4446,  2.9735, -2.8753, -4.9103,  3.5515,
    -2.7649, -2.6436, -4.3281,  2.5358,  3.5522,  4.5746,  4.1062, -2.9213,
    -3.0822, -3.3742, -4.2833, -5.5421,  6.3528, -4.8305,  5.6611,  2.0356,
    -7.4132,  3.7999, -3.5032, -3.3199, -3.5884,  2.5048,  2.0585,  3.9070,
    3.1087, -3.2752, -2.2359, -2.3044,  3.9638, -4.0015, -4.7358, -2.4385,
    -2.7595, -3.7774,  2.9167,  1.7212, -4.1056, -3.2539,  2.9930,  5.4821,
    -0.9080, -2.2462, -4.1286, -4.9021,  3.9147,  4.2125, -2.2905, -3.8277,
    -6.4583, -6.5561,  3.6854, -3.9306,  4.7830,  2.9543,  4.2764,  4.5996,
    3.7813, -6.9687,  2.8327,  3.1347,  4.5773, -2.9960, -4.5374, -3.7505,
    -3.3444,  2.4878,  2.0533,  3.3565,  3.3972,  0.2619,  3.9793, -2.1340,
    7.2500,  3.1587, -2.5491, -4.4956, -3.7766, -3.7853, -1.7816,  5.5186,
    -3.8985,  3.7777,  3.0188, -2.6505, -2.5709, -2.4483, -4.0524,  5.0084,
    -4.4288,  2.3992,  2.8356, -2.3314,  4.8920, -4.3557,  4.0295,  0.5884,
    6.0574,  3.9552,  3.7538, -3.3563, -4.4789, -3.2191, -3.3670,  3.3826,
    -3.6409, -2.4180, -4.3946, -3.8463,  3.9004,  3.2409,  5.2439, -5.1737,
    4.3380,  3.6170,  4.6401,  3.9497, -0.9446, -3.9899, -3.6745,  2.6914,
    3.9964, -3.7445, -2.9309,  3.7317, -5.2299, -3.4732,  6.6469,  2.0490,
    3.2155, -4.0440, -3.5999, -5.6567, -3.9750, -4.3316, -1.9244, -3.7197,
    -2.0258,  1.3506,  4.5243,  3.0692,  2.3486, -5.3424,  3.1627,  3.3108,
    -3.0265,  4.0222, -3.4738, -4.7294, -3.0013,  5.3251, -2.9485, -3.4425,
    -2.7915,  4.1921,  5.8269,  3.2703,  3.9392,  3.0464,  2.7072, -4.9971,
    2.9054,  3.7455, -3.4174,  1.8289, -3.2417, -3.5533,  3.0673,  3.9599,
    4.5355,  4.0669, -3.7865,  4.0504, -3.5386, -3.3625, -5.6070,  3.7046,
    -1.6352, -6.0651,  6.1212, -4.4512, -3.9305, -5.3238, -4.2267,  3.4909,
    -3.7372,  3.9657,  4.9555, -3.8802, -4.0168, -4.0818,  3.7406, -2.9608,
    -4.7644, -5.1160, -3.5000, -2.1473,  3.7051,  0.4822,  3.6102,  3.7462,
    3.6404,  3.6188,  4.2176, -5.1682,  3.7856, -4.3844, -2.8997,  4.4067,
    1.7987,  5.3515,  3.3003, -4.1372,  4.1972, -2.2419, -2.3638,  2.4149,
    3.9554,  2.3366,  3.5648,  4.1453, -2.5295, -1.9449,  3.8180, -3.5611,
    -2.9640, -2.2255, -3.3470,  4.9123, -3.0314,  3.7172, -3.2726,  3.6840,
    -3.3065, -2.9610,  2.1028, -3.4938, -3.7109, -5.6763,  1.4516,  5.0051,
    -3.8999, -3.5646, -4.2523, -4.4535, -4.4238,  2.7203,  1.7606,  3.3034]

    bias = 0.8328

    final_residual = LN(pre_norm)

    logit = sum(.*(probe_token, final_residual)) + bias
    # should return  11.4077
    @test logit ≈ 11.4077 atol=1e-4


end
