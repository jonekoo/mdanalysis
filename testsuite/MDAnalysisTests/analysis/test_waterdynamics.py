# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4 fileencoding=utf-8
#
# MDAnalysis --- http://www.MDAnalysis.org
# Copyright (c) 2006-2015 Naveen Michaud-Agrawal, Elizabeth J. Denning, Oliver Beckstein
# and contributors (see AUTHORS for the full list)
#
# Released under the GNU Public Licence, v2 or any higher version
#
# Please cite your use of MDAnalysis in published work:
#
# N. Michaud-Agrawal, E. J. Denning, T. B. Woolf, and O. Beckstein.
# MDAnalysis: A Toolkit for the Analysis of Molecular Dynamics Simulations.
# J. Comput. Chem. 32 (2011), 2319--2327, doi:10.1002/jcc.21787
#
from __future__ import print_function
import MDAnalysis
import MDAnalysis.analysis.waterdynamics as mdawd

from numpy.testing import TestCase, assert_equal, dec

from MDAnalysisTests.datafiles import waterPSF, waterDCD
from MDAnalysisTests import parser_not_found


class TestWaterdynamics(TestCase):
    @dec.skipif(parser_not_found('DCD'),
                'DCD parser not available. Are you using python 3?')
    def setUp(self):
        self.universe = MDAnalysis.Universe(waterPSF, waterDCD)
        self.selection1 = "byres name OH2"
        self.selection2 = self.selection1

    def test_HydrogenBondLifetimes(self):
        hbl = mdawd.HydrogenBondLifetimes(
            self.universe, self.selection1, self.selection2, 0, 5, 3)
        hbl.run(quiet=True)
        assert_equal(round(hbl.timeseries[2][1], 5), 0.75)

    def test_WaterOrientationalRelaxation(self):
        wor = mdawd.WaterOrientationalRelaxation(
            self.universe, self.selection1, 0, 5, 2)
        wor.run(quiet=True)
        assert_equal(round(wor.timeseries[1][2], 5), 0.45902)

    def test_WaterOrientationalRelaxation_dtmin(self):
        wor = mdawd.WaterOrientationalRelaxation(
            self.universe, self.selection1, 0, 5, 2, dtmin=2)
        wor.run(quiet=True)
        assert_equal(round(wor.timeseries[0][2], 5), 0.45902)

    def test_WaterOrientationalRelaxation_prefetch(self):
        wor = mdawd.WaterOrientationalRelaxation(
            self.universe, self.selection1, 0, 5, 2, prefetch=False)
        wor.run(quiet=True)
        assert_equal(round(wor.timeseries[1][2], 5), 0.45902)

    def test_WaterOrientationalRelaxation_bulk(self):
        wor = mdawd.WaterOrientationalRelaxation(
            self.universe, self.selection1, 0, 5, 2, bulk=True)
        wor.run(quiet=True)
        print(wor.timeseries)
        assert_equal(round(wor.timeseries[1][2], 5), 0.45902)

    def test_WaterOrientationalRelaxation_single(self):
        wor = mdawd.WaterOrientationalRelaxation(
            self.universe, self.selection1, 0, 5, 2, bulk=True, single=False)
        wor.run(quiet=True)
        worsingle = mdawd.WaterOrientationalRelaxation(
           self.universe, self.selection1, 0, 5, 2, bulk=True, single=True)
        worsingle.run(quiet=True)
        for i in range(len(wor.timeseries)):
            for j in range(len(wor.timeseries[0][:])):
                assert_equal(round(wor.timeseries[i][j], 5),
                             round(worsingle.timeseries[i][j], 5))
        print(worsingle.timeseries)
        # assert_equal(round(wor.timeseries[1][2], 5), 0.45902)

    def test_AngularDistribution(self):
        ad = mdawd.AngularDistribution(
            self.universe, self.selection1, 40)
        ad.run(quiet=True)
        assert_equal(str(ad.graph[0][39]), str("0.951172947884 0.48313682125"))

    def test_MeanSquareDisplacement(self):
        msd = mdawd.MeanSquareDisplacement(
            self.universe, self.selection1, 0, 10, 2)
        msd.run(quiet=True)
        assert_equal(round(msd.timeseries[1], 5), 0.03984)

    def test_SurvivalProbability(self):
        sp = mdawd.SurvivalProbability(
            self.universe, self.selection1, 0, 6, 3)
        sp.run(quiet=True)
        assert_equal(round(sp.timeseries[1], 5), 1.0)
