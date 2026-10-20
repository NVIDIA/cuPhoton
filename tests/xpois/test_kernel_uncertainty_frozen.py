# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Frozen references for the batched kernel-fit and correlation code.

The digests below (shape, sum, sum of squares, first values) were recorded
from the per-coefficient and per-position implementations on three seeds,
and the whole arrays in ``FROZEN`` from the batched implementation at the
full linear FFT size. The code must reproduce the digests to ``rtol=1e-12``
and every element to ``rtol=1e-10`` (the padded FFT lengths change the
rounding), with the chunk budgets forced small so several chunks run on
these small inputs, including one coefficient chunk that straddles the
basis and background columns.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from scipy.ndimage import correlate
from scipy.signal import fftconvolve

from cuphoton.xpois import (
    GaussianBasisComponent,
    correlation_factor,
    kernel_fit_covariance,
    kernel_fit_variance,
    psf_weights,
    solve_constant_kernel,
)
from cuphoton.xpois import correlation as correlation_module
from cuphoton.xpois import kernel_uncertainty as kernel_module

SIZE = 96
KERNEL_SHAPE = (9, 9)
COMPONENTS = (
    GaussianBasisComponent(sigma=1.0, degree=1),
    GaussianBasisComponent(sigma=2.0, degree=0),
)
BACKGROUND_DEGREE = 1
N_POSITIONS = 25
WINDOW = 7
SEEDS = (0, 1, 2)
HEAD = 4

# Digests recorded from the per-coefficient and per-position loops.
REFERENCES = {
    "plain": {
        0: {
            "shape": (7, 7),
            "sum": 0.008944378552227728,
            "sumsq": 3.331566958042317e-05,
            "head": [
                3.474726752254163e-09,
                -7.405398857407167e-12,
                -1.2406688318240667e-11,
                -1.125536641426417e-09,
            ],
        },
        1: {
            "shape": (7, 7),
            "sum": 0.008900418136507342,
            "sumsq": 3.24789030743648e-05,
            "head": [
                2.86318538362839e-09,
                -1.3907776609311413e-11,
                8.434946626862118e-12,
                -9.389030757991924e-10,
            ],
        },
        2: {
            "shape": (7, 7),
            "sum": 0.0089155078585706,
            "sumsq": 3.243934633933451e-05,
            "head": [
                2.911950782492047e-09,
                1.5016282833961205e-11,
                4.855975415945293e-12,
                -9.408812333465172e-10,
            ],
        },
    },
    "sandwich": {
        0: {
            "shape": (7, 7),
            "sum": 0.027199218725802728,
            "sumsq": 0.00030599647890609957,
            "head": [
                6.07609766172597e-09,
                -4.519329777650764e-12,
                -2.3529632109063068e-11,
                -2.066691270601429e-09,
            ],
        },
        1: {
            "shape": (7, 7),
            "sum": 0.027083820823675647,
            "sumsq": 0.00029971168754152754,
            "head": [
                5.197629621907343e-09,
                -3.194600736864017e-11,
                1.6074374802060547e-12,
                -1.7955930601554737e-09,
            ],
        },
        2: {
            "shape": (7, 7),
            "sum": 0.027080682498045974,
            "sumsq": 0.00029991860649854565,
            "head": [
                5.232426201476728e-09,
                1.7330138011316347e-11,
                1.4183712743482226e-11,
                -1.7930780963567773e-09,
            ],
        },
    },
    "variance": {
        0: {
            "shape": (25,),
            "sum": 0.009800640796435035,
            "sumsq": 9.80820237146057e-06,
            "head": [
                0.0010886278766617289,
                5.100830812651617e-06,
                3.3555521479633563e-06,
                0.0017308386874634158,
            ],
        },
        1: {
            "shape": (25,),
            "sum": 0.006778761896600701,
            "sumsq": 1.4217631153292737e-05,
            "head": [
                0.00046295539115736244,
                1.4373313095881125e-05,
                3.3483312134536702e-06,
                2.0493338229832e-06,
            ],
        },
        2: {
            "shape": (25,),
            "sum": 0.0010849524677412922,
            "sumsq": 2.3765687678343244e-07,
            "head": [
                1.069538067181868e-05,
                0.00010764235573352007,
                5.7104743432997e-06,
                4.052088957870147e-06,
            ],
        },
    },
    "factor": {
        0: {
            "shape": (25,),
            "sum": 37.454620080088006,
            "sumsq": 56.14260207784114,
            "head": [
                1.504780112222151,
                1.4877896444076137,
                1.4868005934452486,
                1.5394371409990388,
            ],
        },
        1: {
            "shape": (25,),
            "sum": 37.18347335618376,
            "sumsq": 55.31805143951073,
            "head": [
                1.4802190908362907,
                1.486097311551712,
                1.510193970699635,
                1.4975300631714288,
            ],
        },
        2: {
            "shape": (25,),
            "sum": 37.09018981529562,
            "sumsq": 55.04352206616343,
            "head": [
                1.4963515169321016,
                1.5237294841406688,
                1.4059730395770362,
                1.5037407399830267,
            ],
        },
    },
}

# Whole output arrays, flattened, per name and seed.
FROZEN = {
    "plain": {
        0: [
            np.float64(3.474726752254163e-09),
            np.float64(-7.405398857407167e-12),
            np.float64(-1.2406688318240667e-11),
            np.float64(-1.125536641426417e-09),
            np.float64(5.576211442086821e-07),
            np.float64(-2.563165461933023e-07),
            np.float64(2.919484204955173e-07),
            np.float64(-7.405398857407162e-12),
            np.float64(1.8440782549083383e-09),
            np.float64(-2.1009075434957473e-11),
            np.float64(-6.018160055056126e-13),
            np.float64(4.664608235812979e-09),
            np.float64(-1.1994756657854478e-07),
            np.float64(-1.1862189700905041e-09),
            np.float64(-1.2406688318240662e-11),
            np.float64(-2.1009075434957467e-11),
            np.float64(1.7372965560288374e-09),
            np.float64(4.435429507664159e-12),
            np.float64(-4.952932425232573e-09),
            np.float64(1.4887607465331219e-08),
            np.float64(-1.1815859542954248e-07),
            np.float64(-1.1255366414264169e-09),
            np.float64(-6.018160055056118e-13),
            np.float64(4.435429507664161e-12),
            np.float64(3.896083927218425e-10),
            np.float64(-2.4554546956843486e-07),
            np.float64(1.1207881855981438e-07),
            np.float64(-1.2862405837222508e-07),
            np.float64(5.57621144208681e-07),
            np.float64(4.66460823581298e-09),
            np.float64(-4.95293242523257e-09),
            np.float64(-2.4554546956843455e-07),
            np.float64(0.0010126376944111697),
            np.float64(-0.00011441174712844615),
            np.float64(0.00013492744534007343),
            np.float64(-2.5631654619330237e-07),
            np.float64(-1.1994756657854478e-07),
            np.float64(1.488760746533122e-08),
            np.float64(1.1207881855981441e-07),
            np.float64(-0.00011441174712844624),
            np.float64(0.004009238316687794),
            np.float64(-6.837202893036569e-05),
            np.float64(2.919484204955171e-07),
            np.float64(-1.186218970090504e-09),
            np.float64(-1.181585954295425e-07),
            np.float64(-1.28624058372225e-07),
            np.float64(0.00013492744534007348),
            np.float64(-6.837202893036566e-05),
            np.float64(0.004017997143481811),
        ],
        1: [
            np.float64(2.86318538362839e-09),
            np.float64(-1.3907776609311413e-11),
            np.float64(8.434946626862118e-12),
            np.float64(-9.389030757991924e-10),
            np.float64(4.872747386115144e-07),
            np.float64(1.457768852112273e-07),
            np.float64(-1.102732152363375e-07),
            np.float64(-1.3907776609311418e-11),
            np.float64(1.5552462992464424e-09),
            np.float64(-7.270983872213623e-11),
            np.float64(4.775312144223103e-12),
            np.float64(-2.2491987123906114e-09),
            np.float64(-1.0645620299022739e-07),
            np.float64(5.184269475097606e-11),
            np.float64(8.434946626862116e-12),
            np.float64(-7.270983872213621e-11),
            np.float64(1.4753628391391893e-09),
            np.float64(-2.89784419923539e-12),
            np.float64(1.6648648921903849e-09),
            np.float64(3.489817311784136e-09),
            np.float64(-9.83315712748312e-08),
            np.float64(-9.389030757991926e-10),
            np.float64(4.775312144223104e-12),
            np.float64(-2.897844199235391e-12),
            np.float64(3.3036343518233696e-10),
            np.float64(-2.1824588015047095e-07),
            np.float64(-6.585969216349429e-08),
            np.float64(4.9567626348357583e-08),
            np.float64(4.872747386115147e-07),
            np.float64(-2.249198712390612e-09),
            np.float64(1.6648648921903859e-09),
            np.float64(-2.1824588015047103e-07),
            np.float64(0.0009903010160368087),
            np.float64(6.924220358643143e-05),
            np.float64(-5.534286394137832e-05),
            np.float64(1.457768852112273e-07),
            np.float64(-1.064562029902274e-07),
            np.float64(3.489817311784136e-09),
            np.float64(-6.585969216349428e-08),
            np.float64(6.924220358643143e-05),
            np.float64(0.003973506589096912),
            np.float64(-2.6360951727661593e-05),
            np.float64(-1.1027321523633753e-07),
            np.float64(5.184269475097713e-11),
            np.float64(-9.83315712748312e-08),
            np.float64(4.956762634835759e-08),
            np.float64(-5.5342863941378323e-05),
            np.float64(-2.6360951727661593e-05),
            np.float64(0.00396135674176835),
        ],
        2: [
            np.float64(2.911950782492047e-09),
            np.float64(1.5016282833961205e-11),
            np.float64(4.855975415945293e-12),
            np.float64(-9.408812333465172e-10),
            np.float64(4.6140044449951e-07),
            np.float64(-8.00368726186932e-08),
            np.float64(7.657359933720319e-08),
            np.float64(1.5016282833961205e-11),
            np.float64(1.7380830255075144e-09),
            np.float64(4.053196746026185e-12),
            np.float64(-2.9309162516853765e-12),
            np.float64(1.244834716582083e-09),
            np.float64(-1.1302823106709011e-07),
            np.float64(3.825761051562743e-09),
            np.float64(4.855975415945287e-12),
            np.float64(4.053196746026185e-12),
            np.float64(1.6430272945226288e-09),
            np.float64(8.724804975389248e-13),
            np.float64(-2.5841126165316716e-09),
            np.float64(4.69517904241719e-09),
            np.float64(-1.0817726053579814e-07),
            np.float64(-9.408812333465172e-10),
            np.float64(-2.930916251685376e-12),
            np.float64(8.724804975389234e-13),
            np.float64(3.2861706676054733e-10),
            np.float64(-2.1311689680802254e-07),
            np.float64(3.590146051239792e-08),
            np.float64(-3.6298176480765493e-08),
            np.float64(4.6140044449950995e-07),
            np.float64(1.2448347165820825e-09),
            np.float64(-2.584112616531672e-09),
            np.float64(-2.1311689680802254e-07),
            np.float64(0.0009958672560721278),
            np.float64(-4.244057867534091e-05),
            np.float64(4.0243174817165586e-05),
            np.float64(-8.00368726186932e-08),
            np.float64(-1.130282310670901e-07),
            np.float64(4.695179042417189e-09),
            np.float64(3.590146051239793e-08),
            np.float64(-4.244057867534091e-05),
            np.float64(0.003967969418093058),
            np.float64(-2.9058085425279807e-06),
            np.float64(7.657359933720314e-08),
            np.float64(3.825761051562743e-09),
            np.float64(-1.0817726053579815e-07),
            np.float64(-3.6298176480765473e-08),
            np.float64(4.024317481716558e-05),
            np.float64(-2.90580854252798e-06),
            np.float64(0.003961812026099015),
        ],
    },
    "sandwich": {
        0: [
            np.float64(6.07609766172597e-09),
            np.float64(-4.519329777650764e-12),
            np.float64(-2.3529632109063068e-11),
            np.float64(-2.066691270601429e-09),
            np.float64(1.1781583536194367e-06),
            np.float64(-5.060394400814596e-07),
            np.float64(5.786303540686104e-07),
            np.float64(-4.519329777650981e-12),
            np.float64(3.539073697626586e-09),
            np.float64(5.157580038156493e-11),
            np.float64(-7.958164580055015e-13),
            np.float64(4.013756154444054e-09),
            np.float64(-2.1934802392991207e-07),
            np.float64(-7.329610434066314e-09),
            np.float64(-2.3529632109062858e-11),
            np.float64(5.157580038156492e-11),
            np.float64(3.414621243246069e-09),
            np.float64(7.394983617304875e-12),
            np.float64(-7.017277126466107e-09),
            np.float64(1.1854412850828653e-08),
            np.float64(-2.1835439779102645e-07),
            np.float64(-2.0666912706014133e-09),
            np.float64(-7.958164580055825e-13),
            np.float64(7.394983617304881e-12),
            np.float64(7.676054040108619e-10),
            np.float64(-5.63783101354248e-07),
            np.float64(2.4284838450531073e-07),
            np.float64(-2.798149967587669e-07),
            np.float64(1.178158353619433e-06),
            np.float64(4.01375615444412e-09),
            np.float64(-7.017277126466115e-09),
            np.float64(-5.637831013542502e-07),
            np.float64(0.0030170205406142257),
            np.float64(-0.0002612122510695556),
            np.float64(0.00032090624743069847),
            np.float64(-5.060394400814581e-07),
            np.float64(-2.19348023929912e-07),
            np.float64(1.185441285082868e-08),
            np.float64(2.428483845053117e-07),
            np.float64(-0.0002612122510695556),
            np.float64(0.012153609657951269),
            np.float64(-0.00014478804347102146),
            np.float64(5.786303540686075e-07),
            np.float64(-7.329610434066264e-09),
            np.float64(-2.1835439779102645e-07),
            np.float64(-2.798149967587674e-07),
            np.float64(0.0003209062474306983),
            np.float64(-0.00014478804347102135),
            np.float64(0.012198339260362068),
        ],
        1: [
            np.float64(5.197629621907343e-09),
            np.float64(-3.194600736864017e-11),
            np.float64(1.6074374802060547e-12),
            np.float64(-1.7955930601554737e-09),
            np.float64(1.0828668382320407e-06),
            np.float64(3.332997023272515e-07),
            np.float64(-2.4137921192393984e-07),
            np.float64(-3.194600736864077e-11),
            np.float64(2.9918431793333433e-09),
            np.float64(-1.681120138539172e-10),
            np.float64(1.0846524637414349e-11),
            np.float64(-7.407144409268977e-09),
            np.float64(-1.814275528077298e-07),
            np.float64(1.1006094722232772e-08),
            np.float64(1.6074374802061298e-12),
            np.float64(-1.681120138539171e-10),
            np.float64(2.841932472038888e-09),
            np.float64(-2.92719634324493e-13),
            np.float64(3.4594939565466192e-09),
            np.float64(-4.177784342434506e-11),
            np.float64(-1.6912453109122168e-07),
            np.float64(-1.7955930601554671e-09),
            np.float64(1.0846524637414121e-11),
            np.float64(-2.9271963432444695e-13),
            np.float64(6.788516604194648e-10),
            np.float64(-5.210893044978225e-07),
            np.float64(-1.5998110454317382e-07),
            np.float64(1.1568799293515552e-07),
            np.float64(1.082866838232043e-06),
            np.float64(-7.407144409268876e-09),
            np.float64(3.459493956546611e-09),
            np.float64(-5.210893044978244e-07),
            np.float64(0.002967006441581122),
            np.float64(0.00017325190942284963),
            np.float64(-0.00012852608432668518),
            np.float64(3.3329970232725153e-07),
            np.float64(-1.8142755280772984e-07),
            np.float64(-4.177784342435044e-11),
            np.float64(-1.5998110454317413e-07),
            np.float64(0.00017325190942284953),
            np.float64(0.012065678359435094),
            np.float64(-4.501548204919512e-05),
            np.float64(-2.4137921192394e-07),
            np.float64(1.1006094722232762e-08),
            np.float64(-1.691245310912216e-07),
            np.float64(1.156879929351558e-07),
            np.float64(-0.00012852608432668516),
            np.float64(-4.501548204919509e-05),
            np.float64(0.012051175854298123),
        ],
        2: [
            np.float64(5.232426201476728e-09),
            np.float64(1.7330138011316347e-11),
            np.float64(1.4183712743482226e-11),
            np.float64(-1.7930780963567773e-09),
            np.float64(1.0570195223124348e-06),
            np.float64(-1.5596809394833576e-07),
            np.float64(1.7517503549996263e-07),
            np.float64(1.7330138011316117e-11),
            np.float64(3.3234665719302885e-09),
            np.float64(1.7274768905965296e-11),
            np.float64(-4.760650842832324e-12),
            np.float64(1.9975824065418244e-09),
            np.float64(-2.0671977291752685e-07),
            np.float64(8.502821366943788e-11),
            np.float64(1.4183712743482204e-11),
            np.float64(1.7274768905965296e-11),
            np.float64(3.056091132276061e-09),
            np.float64(-1.982999968140637e-12),
            np.float64(-2.5321049831672885e-09),
            np.float64(2.4991550919046257e-09),
            np.float64(-1.8468758652910318e-07),
            np.float64(-1.7930780963567734e-09),
            np.float64(-4.760650842832382e-12),
            np.float64(-1.9829999681406657e-12),
            np.float64(6.7948900317654e-10),
            np.float64(-5.262213243464805e-07),
            np.float64(7.840244192052308e-08),
            np.float64(-8.789730300798765e-08),
            np.float64(1.0570195223124297e-06),
            np.float64(1.9975824065418306e-09),
            np.float64(-2.532104983167294e-09),
            np.float64(-5.262213243464799e-07),
            np.float64(0.003003145246996927),
            np.float64(-9.165582824408186e-05),
            np.float64(0.00010877224128784346),
            np.float64(-1.5596809394833433e-07),
            np.float64(-2.0671977291752677e-07),
            np.float64(2.499155091904616e-09),
            np.float64(7.84024419205228e-08),
            np.float64(-9.165582824408177e-05),
            np.float64(0.01204831264469582),
            np.float64(-3.785959954024774e-05),
            np.float64(1.7517503549996154e-07),
            np.float64(8.50282136694354e-11),
            np.float64(-1.846875865291032e-07),
            np.float64(-8.789730300798749e-08),
            np.float64(0.00010877224128784345),
            np.float64(-3.785959954024776e-05),
            np.float64(0.01207039988478012),
        ],
    },
    "variance": {
        0: [
            np.float64(0.0010886278766617289),
            np.float64(5.100830812651617e-06),
            np.float64(3.3555521479633563e-06),
            np.float64(0.0017308386874634158),
            np.float64(0.0007696356970614365),
            np.float64(2.786283649219833e-06),
            np.float64(3.9693607830897804e-05),
            np.float64(0.0007988129679830468),
            np.float64(4.895196803813303e-06),
            np.float64(0.0013468070701323348),
            np.float64(0.0010058997517390716),
            np.float64(1.043252650586943e-05),
            np.float64(0.00012554931104312488),
            np.float64(0.0007009194650042793),
            np.float64(7.846360311391372e-06),
            np.float64(0.000433846510509737),
            np.float64(0.0004813134954158041),
            np.float64(2.3705209073352973e-06),
            np.float64(0.0005352074239462093),
            np.float64(4.851300435673955e-06),
            np.float64(4.1005070144967274e-06),
            np.float64(1.1734468714141867e-05),
            np.float64(9.093511678306005e-05),
            np.float64(0.0005894416535236164),
            np.float64(5.638614034716728e-06),
        ],
        1: [
            np.float64(0.00046295539115736244),
            np.float64(1.4373313095881125e-05),
            np.float64(3.3483312134536702e-06),
            np.float64(2.0493338229832e-06),
            np.float64(5.435385423955022e-05),
            np.float64(0.0031366162140929205),
            np.float64(0.0012848057925194689),
            np.float64(1.2931716798192049e-05),
            np.float64(6.938350312433376e-06),
            np.float64(6.277238762677285e-06),
            np.float64(5.658666386249193e-06),
            np.float64(4.73871786103816e-06),
            np.float64(4.557185837190173e-06),
            np.float64(1.7390168547215065e-05),
            np.float64(3.0475084117738173e-06),
            np.float64(1.1496046673371781e-05),
            np.float64(6.540721057323377e-06),
            np.float64(6.2117649348059345e-06),
            np.float64(4.759798562922762e-06),
            np.float64(0.00012008678111109292),
            np.float64(7.448621218671332e-06),
            np.float64(1.1386684391023697e-05),
            np.float64(4.312652611118231e-06),
            np.float64(6.7645123173881875e-06),
            np.float64(0.0015797125306645928),
        ],
        2: [
            np.float64(1.069538067181868e-05),
            np.float64(0.00010764235573352007),
            np.float64(5.7104743432997e-06),
            np.float64(4.052088957870147e-06),
            np.float64(2.9230411418976756e-06),
            np.float64(1.260790666991094e-05),
            np.float64(2.700256196916115e-05),
            np.float64(3.931745946402238e-06),
            np.float64(0.00030515583232937745),
            np.float64(1.7922579490755808e-06),
            np.float64(1.0200847725768235e-05),
            np.float64(3.4466803565924672e-06),
            np.float64(2.1208890608126273e-06),
            np.float64(2.440433411460242e-06),
            np.float64(9.709663638892026e-06),
            np.float64(7.080455953479349e-06),
            np.float64(2.798533856890036e-06),
            np.float64(1.8744303344047199e-06),
            np.float64(9.932703210453624e-06),
            np.float64(0.00019057880546329006),
            np.float64(7.727165158412936e-06),
            np.float64(3.104500157101495e-05),
            np.float64(1.1021487695790226e-05),
            np.float64(0.0003065574258797327),
            np.float64(6.904298711964213e-06),
        ],
    },
    "factor": {
        0: [
            np.float64(1.504780112222151),
            np.float64(1.4877896444076137),
            np.float64(1.4868005934452486),
            np.float64(1.5394371409990388),
            np.float64(1.5089778800682132),
            np.float64(1.4515591553422127),
            np.float64(1.4814447052741864),
            np.float64(1.486524310460637),
            np.float64(1.4952260262691925),
            np.float64(1.4913796511945634),
            np.float64(1.5219433198462946),
            np.float64(1.5026082296004748),
            np.float64(1.5347773988794036),
            np.float64(1.4647447390066004),
            np.float64(1.5058966232903803),
            np.float64(1.5554819063348608),
            np.float64(1.535574178582861),
            np.float64(1.4435509040906203),
            np.float64(1.5079203589901082),
            np.float64(1.5154925480614698),
            np.float64(1.4007205281995414),
            np.float64(1.498520771611894),
            np.float64(1.4769239904658737),
            np.float64(1.5505435025521708),
            np.float64(1.5060018608924002),
        ],
        1: [
            np.float64(1.4802190908362907),
            np.float64(1.486097311551712),
            np.float64(1.510193970699635),
            np.float64(1.4975300631714288),
            np.float64(1.4970405260020034),
            np.float64(1.4489714564424072),
            np.float64(1.5007338440233498),
            np.float64(1.4983116875690572),
            np.float64(1.4652186384447807),
            np.float64(1.50234555542707),
            np.float64(1.4913871073818539),
            np.float64(1.460542354691985),
            np.float64(1.5471446857472813),
            np.float64(1.5079068312801673),
            np.float64(1.4894557402825812),
            np.float64(1.4995319814577783),
            np.float64(1.4520026078928068),
            np.float64(1.4524372296240935),
            np.float64(1.4770804083832265),
            np.float64(1.5041421923971376),
            np.float64(1.4801754910232692),
            np.float64(1.5146087552847995),
            np.float64(1.4970087427542282),
            np.float64(1.4761724248550987),
            np.float64(1.4472146589597208),
        ],
        2: [
            np.float64(1.4963515169321016),
            np.float64(1.5237294841406688),
            np.float64(1.4059730395770362),
            np.float64(1.5037407399830267),
            np.float64(1.4660864484018468),
            np.float64(1.4944244062645706),
            np.float64(1.515262126251138),
            np.float64(1.4751983336582712),
            np.float64(1.4604340507060107),
            np.float64(1.492774108871634),
            np.float64(1.4733532180769633),
            np.float64(1.499146852671858),
            np.float64(1.453002956862662),
            np.float64(1.4826578413020166),
            np.float64(1.4738459325619215),
            np.float64(1.4831645683393286),
            np.float64(1.5008743728530698),
            np.float64(1.4971679576015813),
            np.float64(1.4663307450429464),
            np.float64(1.4927900790813005),
            np.float64(1.4529396758445112),
            np.float64(1.5028787762704496),
            np.float64(1.461939851844013),
            np.float64(1.526643563666574),
            np.float64(1.4894791684901312),
        ],
    },
}


def _gaussian(shape, sigma):
    y = np.arange(shape[0]) - shape[0] // 2
    x = np.arange(shape[1]) - shape[1] // 2
    k = np.exp(-0.5 * (y[:, None] ** 2 + x[None, :] ** 2) / sigma**2)
    return k / k.sum()


def _resampling_stamp():
    taps = np.array([0.1, 0.4, 0.4, 0.1])
    return taps[:, None] * taps[None, :]


def scene(seed):
    """A fitted scene with noise, variance planes, positions and weights."""

    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    raw = np.full((SIZE, SIZE), 10.0)
    for x0, y0 in rng.uniform(20, SIZE - 20, (12, 2)):
        raw += 3000.0 * np.exp(
            -0.5 * ((xx - x0) ** 2 + (yy - y0) ** 2) / 1.5**2
        )
    stamp = _resampling_stamp()
    reference = correlate(raw, stamp, mode="nearest")
    reference += rng.normal(0.0, 1.0, reference.shape)
    target = fftconvolve(reference, _gaussian(KERNEL_SHAPE, 1.2), "same")
    target += 2.0 + 0.01 * xx + rng.normal(0.0, 2.0, target.shape)
    target_variance = rng.uniform(3.0, 5.0, (SIZE, SIZE))
    reference_variance = rng.uniform(0.8, 1.2, (SIZE, SIZE))
    fit_mask = np.zeros((SIZE, SIZE), dtype=bool)
    fit_mask[12:-12, 12:-12] = True
    fit = solve_constant_kernel(
        reference,
        target,
        COMPONENTS,
        kernel_shape=KERNEL_SHAPE,
        variance=target_variance,
        fit_mask=fit_mask,
        background_degree=BACKGROUND_DEGREE,
        backend="cpu",
    )
    positions = rng.uniform(15, SIZE - 15, (N_POSITIONS, 2))
    weights = psf_weights(
        1.5, WINDOW, offset=positions - np.rint(positions), derivative="x"
    )
    valid = np.ones((SIZE, SIZE), dtype=bool)
    valid[:, 40] = False
    return {
        "fit": fit,
        "reference": reference,
        "target_variance": target_variance,
        "reference_variance": reference_variance,
        "stamp": stamp,
        "positions": positions,
        "weights": weights,
        "valid": valid,
    }


def outputs(seed, xp=np, footprint_chunk=None, response_chunk=None):
    """The four arrays the references digest, for one seed.

    With ``xp`` the inputs (and the fit's arrays) move to that namespace;
    the chunk budgets are passed through to the two calls.
    """

    s = scene(seed)
    if xp is not np:
        s = {
            key: xp.asarray(value) for key, value in s.items() if key != "fit"
        }
        fit = scene(seed)["fit"]
        s["fit"] = replace(
            fit,
            basis_kernels=xp.asarray(fit.basis_kernels),
            fit_mask=xp.asarray(fit.fit_mask),
            kernel=xp.asarray(fit.kernel),
        )
    plain = kernel_fit_covariance(
        s["fit"],
        reference=s["reference"],
        target_variance=s["target_variance"],
    )
    sandwich = kernel_fit_covariance(
        s["fit"],
        reference=s["reference"],
        target_variance=s["target_variance"],
        reference_variance=s["reference_variance"],
        resampling_weights=s["stamp"],
        chunk_elements=footprint_chunk,
    )
    variance = kernel_fit_variance(
        sandwich,
        reference=s["reference"],
        positions=s["positions"],
        weights=s["weights"],
        chunk_elements=response_chunk,
    )
    factor = correlation_factor(
        s["weights"],
        kernel=s["fit"].kernel,
        target_variance=s["target_variance"],
        reference_variance=s["reference_variance"],
        resampling_weights=s["stamp"],
        positions=s["positions"],
        valid=s["valid"],
    ).factor
    return {
        "plain": plain.covariance,
        "sandwich": sandwich.covariance,
        "variance": variance,
        "factor": factor,
    }


def digest(array):
    flat = np.asarray(array, dtype=np.float64).ravel()
    return {
        "shape": tuple(np.shape(array)),
        "sum": float(flat.sum()),
        "sumsq": float((flat**2).sum()),
        "head": [float(v) for v in flat[:HEAD]],
    }


def _assert_digest(array, expected):
    got = digest(array)
    assert got["shape"] == tuple(expected["shape"])
    np.testing.assert_allclose(got["sum"], expected["sum"], rtol=1e-12)
    np.testing.assert_allclose(got["sumsq"], expected["sumsq"], rtol=1e-12)
    np.testing.assert_allclose(got["head"], expected["head"], rtol=1e-12)


def _assert_frozen(name, seed, array):
    _assert_digest(array, REFERENCES[name][seed])
    expected = np.reshape(FROZEN[name][seed], REFERENCES[name][seed]["shape"])
    np.testing.assert_allclose(np.asarray(array), expected, rtol=1e-10)


@pytest.mark.parametrize("seed", SEEDS)
def test_batched_code_reproduces_the_frozen_references(seed):
    for name, array in outputs(seed).items():
        _assert_frozen(name, seed, array)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("footprint_budget", [2 * SIZE**2, 3 * SIZE**2])
def test_small_chunks_reproduce_the_frozen_references(
    seed, footprint_budget, monkeypatch
):
    """Several coefficient and position chunks give the one-chunk result.

    The budget of three images makes the middle coefficient chunk straddle
    the last basis column and the first background column.
    """

    monkeypatch.setattr(
        kernel_module, "_FOOTPRINT_CHUNK_ELEMENTS", footprint_budget
    )
    monkeypatch.setattr(kernel_module, "_RESPONSE_CHUNK_ELEMENTS", 1)
    monkeypatch.setattr(correlation_module, "_CHUNK", 7)
    for name, array in outputs(seed).items():
        _assert_frozen(name, seed, array)


@pytest.mark.parametrize("seed", SEEDS)
def test_chunk_elements_pads_the_last_chunk_to_the_same_result(seed):
    """Budgets that leave a partial last chunk (7 coefficients in chunks of
    2; 25 positions in chunks of 4) reproduce the frozen references."""

    n_basis = int(scene(seed)["fit"].basis_kernels.shape[0])
    response = 4 * n_basis * (WINDOW + 2 * KERNEL_SHAPE[0] - 2) ** 2
    for name, array in outputs(
        seed, footprint_chunk=2 * SIZE**2, response_chunk=response
    ).items():
        _assert_frozen(name, seed, array)


@pytest.mark.parametrize("seed", SEEDS)
def test_device_inputs_reproduce_the_frozen_references(seed):
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.runtime.getDeviceCount()
    except Exception as exc:  # pragma: no cover - no CUDA runtime
        pytest.skip(f"CuPy CUDA runtime is not usable: {exc}")
    for name, array in outputs(seed, xp=cp).items():
        assert type(array).__module__ == "cupy"
        _assert_frozen(name, seed, cp.asnumpy(array))
