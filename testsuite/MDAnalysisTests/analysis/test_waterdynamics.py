# -*- Mode: python; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4 fileencoding=utf-8
#
# MDAnalysis --- http://www.mdanalysis.org
# Copyright (c) 2006-2017 The MDAnalysis Development Team and contributors
# (see the file AUTHORS for the full list of names)
#
# Released under the GNU Public Licence, v2 or any higher version
#
# Please cite your use of MDAnalysis in published work:
#
# R. J. Gowers, M. Linke, J. Barnoud, T. J. E. Reddy, M. N. Melo, S. L. Seyler,
# D. L. Dotson, J. Domanski, S. Buchoux, I. M. Kenney, and O. Beckstein.
# MDAnalysis: A Python package for the rapid analysis of molecular dynamics
# simulations. In S. Benthall and S. Rostrup editors, Proceedings of the 15th
# Python in Science Conference, pages 102-109, Austin, TX, 2016. SciPy.
#
# N. Michaud-Agrawal, E. J. Denning, T. B. Woolf, and O. Beckstein.
# MDAnalysis: A Toolkit for the Analysis of Molecular Dynamics Simulations.
# J. Comput. Chem. 32 (2011), 2319--2327, doi:10.1002/jcc.21787
#
from __future__ import print_function, absolute_import
import MDAnalysis
import MDAnalysis.analysis.waterdynamics as mdawd
import pytest

from MDAnalysisTests.datafiles import waterPSF, waterDCD, bulkworDCD, \
    tip4p05prmtop, tip4p05bulkworDCD
from numpy.testing import assert_almost_equal
import numpy

SELECTION1 = "byres name OH2"
SELECTION2 = "byres name P1"


@pytest.fixture(scope='module')
def universe():
    return MDAnalysis.Universe(waterPSF, waterDCD)


def test_HydrogenBondLifetimes(universe):
    hbl = MDAnalysis.analysis.waterdynamics.HydrogenBondLifetimes(universe,
                                                                  SELECTION1,
                                                                  SELECTION1,
                                                                  0, 5, 3)
    hbl.run()
    assert_almost_equal(hbl.timeseries[2][1], 0.75, 5)


def test_WaterOrientationalRelaxation(universe):
    wor = MDAnalysis.analysis.waterdynamics.WaterOrientationalRelaxation(
        universe,
        SELECTION1, 0, 5, 2)
    wor.run()
    assert round(wor.timeseries[1][2], 5) == 0.35887


def test_WaterOrientationalRelaxation_zeroMolecules(universe):
    wor_zero = MDAnalysis.analysis.waterdynamics.WaterOrientationalRelaxation(
        universe,
        SELECTION2, 0, 5, 2)
    wor_zero.run()
    assert wor_zero.timeseries[1] == (0.0, 0.0, 0.0)


def test_AngularDistribution(universe):
    ad = MDAnalysis.analysis.waterdynamics.AngularDistribution(universe,
                                                               SELECTION1,
                                                               40)
    ad.run()
    assert str(ad.graph[0][39]) == str("0.951172947884 0.48313682125")


def test_MeanSquareDisplacement(universe):
    msd = MDAnalysis.analysis.waterdynamics.MeanSquareDisplacement(universe,
                                                                   SELECTION1,
                                                                   0, 10, 2)
    msd.run()
    assert round(msd.timeseries[1], 5) == 0.03984


def test_MeanSquareDisplacement_zeroMolecules(universe):
    msd_zero = MDAnalysis.analysis.waterdynamics.MeanSquareDisplacement(
        universe,
        SELECTION2, 0, 10, 2)
    msd_zero.run()
    assert msd_zero.timeseries[1] == 0.0


def test_SurvivalProbability(universe):
    sp = MDAnalysis.analysis.waterdynamics.SurvivalProbability(universe,
                                                               SELECTION1,
                                                               0, 6, 3)
    sp.run()
    assert round(sp.timeseries[1], 5) == 1.0


def test_SurvivalProbability_zeroMolecules(universe):
    sp_zero = MDAnalysis.analysis.waterdynamics.SurvivalProbability(universe,
                                                                    SELECTION2,
                                                                    0, 6, 3)
    sp_zero.run()
    assert sp_zero.timeseries[1] == 0.0


def test_BulkWaterOrientationalRelaxation(universe):
        # Fabricate data in which only a single molecule is rotating at
        # constant angular velocity around its dipole axis.
        water = universe.select_atoms(SELECTION1)
        selection = SELECTION1 + " and resid " + \
            str(water.resids[0]) + "-" + str(water.resids[0])
        w = 0.1  # angular velocity, radians/timestep
        if False:
            # This code can be used to create the bulkworDCD file:
            rO = [0.0, 0.0, 0.0]  # put the oxygen to the origin
            rOH = 0.9572  # bond length, aengstroms
            HOH = 104.52 / 180.0 * numpy.pi  # angle, radians
            Hz = rOH * numpy.cos(HOH / 2)
            Hx = rOH * numpy.sin(HOH / 2)
            water = universe.select_atoms(selection)
            with MDAnalysis.coordinates.DCD.DCDWriter(
                    "bulkwor.dcd",
                    n_atoms=universe.atoms.n_atoms) as writer:
                for i, ts in enumerate(universe.trajectory):
                    newpositions = [
                        rO, [Hx * numpy.cos(w * i), Hx * numpy.sin(w * i), Hz],
                        [-Hx * numpy.cos(w * i), -Hx * numpy.sin(w * i), Hz]]
                    water.positions = newpositions
                    writer.write_next_timestep(ts)
        u = MDAnalysis.Universe(waterPSF, bulkworDCD)

        # Select only the first water molecule and test
        wor = mdawd.BulkWaterOrientationalRelaxation(
            u, selection, 0, len(u.trajectory), len(u.trajectory)-1, bulk=True)
        wor.run()

        # Test the lenght of timeseries:
        assert len(wor.timeseries) == len(u.trajectory)

        # Test correlation at time t=0 for :
        assert_almost_equal(wor.timeseries[0][0], 1.00000, 5)
        assert_almost_equal(wor.timeseries[0][1], 1.00000, 5)
        assert_almost_equal(wor.timeseries[0][2], 1.00000, 5)

        # Correlation at general time t:
        for i, rec in enumerate(wor.timeseries):
            # Dipole vector:
            assert_almost_equal(round(rec[2], 5), 1.0)
            # H-H bond:
            assert_almost_equal(rec[1], 1.5 * numpy.cos(w * i)**2 - 0.5, 5)


def test_BulkWaterOrientationalRelaxation_tip4p(universe):
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
        wor = mdawd.BulkWaterOrientationalRelaxation(
            u, selection, 0, len(u.trajectory), len(u.trajectory)-1,
            bulk=True)
        wor.run()

        # Test the lenght of timeseries:
        assert len(wor.timeseries) == len(u.trajectory)

        # Test correlation at time t=0 for :
        assert_almost_equal(wor.timeseries[0][0], 1.00000, 5)
        assert_almost_equal(wor.timeseries[0][1], 1.00000, 5)
        assert_almost_equal(wor.timeseries[0][2], 1.00000, 5)

        # Correlation at general time t:
        for i, rec in enumerate(wor.timeseries):
            # Dipole vector:
            assert_almost_equal(rec[2], 1.0, 5)
            # H-H bond:
            assert_almost_equal(rec[1], 1.5 * numpy.cos(w * i)**2 - 0.5, 5)


def test_BulkWaterOrientationalRelaxation_single(universe):
        # Fabricate data in which only a single molecule is rotating at
        # constant angular velocity around its dipole axis.
        u = MDAnalysis.Universe(waterPSF, bulkworDCD)

        # Select only the first water molecule and test
        wor = mdawd.BulkWaterOrientationalRelaxation(
            u, SELECTION1, 0, len(u.trajectory), len(u.trajectory)-1,
            bulk=True, single=True)
        wor.run()

        print(len(wor.HHC2s[1]))
        # Test the lenght of timeseries:
        for C2 in wor.HHC2s:
            assert len(C2) == len(u.trajectory)

        # Test correlation at time t=0 for :
        for C2 in wor.HHC2s:
            assert_almost_equal(C2[0], 1.00000, 5)
        for C2 in wor.OHC2s:
            assert_almost_equal(C2[0], 1.00000, 5)
        for C2 in wor.dipC2s:
            assert_almost_equal(C2[0], 1.00000, 5)

        w = 0.1  # angular velocity, radians/timestep
        # Correlation at general time t:
        for i, rec in enumerate(wor.HHC2s[0]):
            # H-H bond:
            assert_almost_equal(rec, 1.5 * numpy.cos(w * i)**2 - 0.5, 5)
        for i, rec in enumerate(wor.dipC2s[0]):
            # Dipole vector:
            assert_almost_equal(rec, 1.0, 5)


# def test_BulkWaterOrientationalRelaxation_dtmin(self):
#     wor = mdawd.BulkWaterOrientationalRelaxation(
#         universe, SELECTION1, 0, 5, 2, dtmin=2)
#     wor.run(quiet=True)
#     assert_equal(round(wor.timeseries[0][2], 5), 0.45902)


def test_BulkWaterOrientationalRelaxation_tip4p_conditional(universe):
        selection1 = 'resname WAT'
        w = 0.1  # angular velocity, radians/timestep
        u = MDAnalysis.Universe(tip4p05prmtop, tip4p05bulkworDCD)
        water = u.select_atoms(selection1)

        # Select only the first water molecule and test
        # selection = selection1 + " and resid " + \
        #    str(water.resids[0]) + "-" + str(water.resids[0])

        wor = mdawd.BulkWaterOrientationalRelaxation(
            u, 'resid ' + str(water.resids[0]) + "-" + str(water.resids[0]),
            0, len(u.trajectory),
            len(u.trajectory)-1, bulk=True, allwater=selection1)
        wor.run()

        # Test the lenght of timeseries:
        assert len(wor.timeseries) == len(u.trajectory)

        # Test correlation at time t=0 for :
        assert_almost_equal(wor.timeseries[0][0], 1.00000, 5)
        assert_almost_equal(wor.timeseries[0][1], 1.00000, 5)
        assert_almost_equal(wor.timeseries[0][2], 1.00000, 5)

        # Correlation at general time t:
        for i, rec in enumerate(wor.timeseries):
            # Dipole vector:
            assert_almost_equal(rec[2], 1.0, 5)
            # H-H bond:
            assert_almost_equal(rec[1], 1.5 * numpy.cos(w * i)**2 - 0.5, 5)
