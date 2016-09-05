# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4 fileencoding=utf-8
#
# MDAnalysis --- http://www.MDAnalysis.org
# Copyright (c) 2006-2015 Naveen Michaud-Agrawal, Elizabeth J. Denning,
# Oliver Beckstein and contributors (see AUTHORS for the full list)
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

from MDAnalysisTests.datafiles import waterPSF, waterDCD, bulkworDCD, \
    tip4p05prmtop, tip4p05bulkworDCD
from MDAnalysisTests import parser_not_found
import numpy


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
        # Fabricate data in which only a single molecule is rotating at
        # constant angular velocity around its dipole axis.
        water = self.universe.select_atoms(self.selection1)
        selection = self.selection1 + " and resid " + \
            str(water.resids[0]) + "-" + str(water.resids[0])
        w = 0.1  # angular velocity, radians/timestep
        if False:
            # This code can be used to create the bulkworDCD file:
            rO = [0.0, 0.0, 0.0]  # put the oxygen to the origin
            rOH = 0.9572  # bond length, aengstroms
            HOH = 104.52 / 180.0 * numpy.pi  # angle, radians
            Hz = rOH * numpy.cos(HOH / 2)
            Hx = rOH * numpy.sin(HOH / 2)
            water = self.universe.select_atoms(selection)
            with MDAnalysis.coordinates.DCD.DCDWriter(
                    "bulkwor.dcd",
                    n_atoms=self.universe.atoms.n_atoms) as writer:
                for i, ts in enumerate(self.universe.trajectory):
                    newpositions = [
                        rO, [Hx * numpy.cos(w * i), Hx * numpy.sin(w * i), Hz],
                        [-Hx * numpy.cos(w * i), -Hx * numpy.sin(w * i), Hz]]
                    water.positions = newpositions
                    writer.write_next_timestep(ts)
        u = MDAnalysis.Universe(waterPSF, bulkworDCD)

        # Select only the first water molecule and test
        wor = MDAnalysis.analysis.waterdynamics.WaterOrientationalRelaxation(
            u, selection, 0, len(u.trajectory), len(u.trajectory)-1,
            bulk=True)
        wor.run()

        # Test the lenght of timeseries:
        assert_equal(len(wor.timeseries), len(u.trajectory))

        # Test correlation at time t=0 for :
        assert_equal(round(wor.timeseries[0][0], 5), 1.00000)
        assert_equal(round(wor.timeseries[0][1], 5), 1.00000)
        assert_equal(round(wor.timeseries[0][2], 5), 1.00000)

        # Correlation at general time t:
        for i, rec in enumerate(wor.timeseries):
            # Dipole vector:
            assert_equal(round(rec[2], 5), 1.0)
            # H-H bond:
            assert_equal(round(rec[1], 5),
                         round(1.5 * numpy.cos(w * i)**2 - 0.5, 5))

    def test_WaterOrientationalRelaxation_bulk_tip4p(self):
        selection1 = 'resname WAT'
        w = 0.1  # angular velocity, radians/timestep
        create = False
        if create:
            # Fabricate data in which only a single molecule is rotating at
            # constant angular velocity around its dipole axis.
            u = MDAnalysis.Universe(tip4p05prmtop, 'tip4p05temp.dcd')
            water = u.select_atoms(selection1)
            selection = selection1 + " and resid " + \
                str(water.resids[0]) + "-" + str(water.resids[0])
            # This code can be used to create the DCD file:
            rO = [0.0, 0.0, 0.0]  # put the oxygen to the origin
            rM = [0, 0, 0.1546]
            rOH = 0.9572  # bond length, aengstroms
            HOH = 104.52 / 180.0 * numpy.pi  # angle, radians
            Hz = rOH * numpy.cos(HOH / 2)
            Hx = rOH * numpy.sin(HOH / 2)
            water = u.select_atoms(selection)
            with MDAnalysis.coordinates.DCD.DCDWriter(
                    tip4p05bulkworDCD,
                    n_atoms=u.atoms.n_atoms) as writer:
                for i, ts in enumerate(u.trajectory):
                    newpositions = [
                        rO, [Hx * numpy.cos(w * i), Hx * numpy.sin(w * i), Hz],
                        [-Hx * numpy.cos(w * i), -Hx * numpy.sin(w * i), Hz],
                        rM]
                    water.positions = newpositions
                    writer.write_next_timestep(ts)
        u = MDAnalysis.Universe(tip4p05prmtop, tip4p05bulkworDCD)
        water = u.select_atoms(selection1)
        selection = selection1 + " and resid " + \
            str(water.resids[0]) + "-" + str(water.resids[0])


        # Select only the first water molecule and test
        wor = MDAnalysis.analysis.waterdynamics.WaterOrientationalRelaxation(
            u, selection, 0, len(u.trajectory), len(u.trajectory)-1,
            bulk=True)
        wor.run()

        # Test the lenght of timeseries:
        assert_equal(len(wor.timeseries), len(u.trajectory))

        # Test correlation at time t=0 for :
        assert_equal(round(wor.timeseries[0][0], 5), 1.00000)
        assert_equal(round(wor.timeseries[0][1], 5), 1.00000)
        assert_equal(round(wor.timeseries[0][2], 5), 1.00000)

        # Correlation at general time t:
        for i, rec in enumerate(wor.timeseries):
            # Dipole vector:
            assert_equal(round(rec[2], 5), 1.0)
            # H-H bond:
            assert_equal(round(rec[1], 5),
                         round(1.5 * numpy.cos(w * i)**2 - 0.5, 5))

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
