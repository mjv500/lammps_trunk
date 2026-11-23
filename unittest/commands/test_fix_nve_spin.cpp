/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS Development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

// tests of the spin integrators of fix nve/spin that hold for any number of
// MPI processes (the test is run in serial and with 4 MPI processes):
//
// - every spin update is a rotation, so the length of each spin direction
//   vector stays 1 to round-off, without renormalization, for all integrators
//   (sequential, sequential with selfconsistent iterations, and midpoint) and
//   with fields that depend on the spin itself (biquadratic exchange,
//   uniaxial and cubic anisotropy), with fix langevin/spin, and on a frozen
//   and a moving lattice; the magnitude of the magnetic moments (sp[i][3])
//   must not change at all
// - selfconsistent 0 is the original sequential update, and the tol keyword
//   only changes converged results by round-off
// - invalid keyword values and combinations are rejected

#include "../testing/core.h"

#include "atom.h"
#include "info.h"
#include "platform.h"

#include "fmt/format.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <map>
#include <mpi.h>
#include <string>
#include <vector>

// whether to print verbose output (i.e. not capturing LAMMPS screen output).
bool verbose = false;

namespace LAMMPS_NS {

// max. allowed deviation of |s_i| from 1 (round-off is ~1e-15 per step)
static constexpr double EPSILON_NORM = 1.0e-12;

// magnitude of the magnetic moments set in build()
static constexpr double MOMENT = 2.2;

class FixNVESpinTest : public LAMMPSTest {
protected:
    void SetUp() override
    {
        testbinary = "FixNVESpinTest";
        LAMMPSTest::SetUp();
        if (!info->has_style("fix", "nve/spin")) GTEST_SKIP() << "SPIN package not available";
    }

    // 6x6x6 bcc iron (432 atoms) with random spins; bilinear and biquadratic exchange,
    // Zeeman field, uniaxial and cubic anisotropy, i.e. fields on each spin that
    // depend on the spin itself; optionally fix langevin/spin.  the sub-domains of a
    // 2x2x1 processor grid are wide enough for the sectoring of the sequential update.
    void build_system()
    {
        BEGIN_HIDE_OUTPUT();
        command("units           metal");
        command("atom_style      spin");
        command("atom_modify     map array");
        command("boundary        p p p");
        command("lattice         bcc 2.8665");
        command("region          box block 0 6 0 6 0 6");
        command("create_box      1 box");
        command("create_atoms    1 box");
        command("mass            1 55.845");
        command(fmt::format("set             group all spin/atom/random 31 {}", MOMENT));
        command("velocity        all create 300.0 4928459 dist gaussian");
        command("pair_style      spin/exchange/biquadratic 3.5");
        command("pair_coeff      * * biquadratic 3.5 0.02726 0.2171 1.841 "
                "0.01 0.2171 1.841 offset no");
        command("neighbor        0.3 bin");
        command("neigh_modify    every 1 delay 0 check yes");
        command("timestep        0.0001");
        END_HIDE_OUTPUT();
    }

    void build(const std::string &nveopt, const std::string &langevin = "")
    {
        build_system();
        BEGIN_HIDE_OUTPUT();
        command("fix             1 all nve/spin " + nveopt);
        command("fix             2 all precession/spin zeeman 10.0 0.0 0.0 1.0 "
                "anisotropy 0.01 0.0 0.0 1.0 "
                "cubic 0.001 0.0005 1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0");
        if (!langevin.empty()) command("fix             3 all langevin/spin " + langevin);
        END_HIDE_OUTPUT();
    }

    // spins (direction and magnitude) of the local atoms by atom ID
    std::map<tagint, std::vector<double>> local_spins()
    {
        std::map<tagint, std::vector<double>> spins;
        const auto *const tag = lmp->atom->tag;
        const auto *const sp  = lmp->atom->sp;
        for (int i = 0; i < lmp->atom->nlocal; ++i)
            spins[tag[i]] = {sp[i][0], sp[i][1], sp[i][2], sp[i][3]};
        return spins;
    }

    // largest deviation of |s_i| from 1 and of the moment magnitude from MOMENT
    // over all atoms
    void norm_errors(double &normerr, double &momerr)
    {
        const auto *const sp = lmp->atom->sp;
        double err[2]        = {0.0, 0.0};
        for (int i = 0; i < lmp->atom->nlocal; ++i) {
            const double norm = sqrt(sp[i][0] * sp[i][0] + sp[i][1] * sp[i][1] + sp[i][2] * sp[i][2]);
            err[0] = std::max(err[0], fabs(norm - 1.0));
            err[1] = std::max(err[1], fabs(sp[i][3] - MOMENT));
        }
        double all[2];
        MPI_Allreduce(err, all, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        normerr = all[0];
        momerr  = all[1];
    }

    // largest change of a spin component of the atoms that stayed local
    double max_spin_change(const std::map<tagint, std::vector<double>> &before)
    {
        const auto after = local_spins();
        double change    = 0.0;
        for (const auto &s : before) {
            const auto it = after.find(s.first);
            if (it == after.end()) continue;
            for (int k = 0; k < 3; ++k)
                change = std::max(change, fabs(it->second[k] - s.second[k]));
        }
        double all = 0.0;
        MPI_Allreduce(&change, &all, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        return all;
    }

    // largest difference of the spins of two runs, compared by atom ID
    // (on the same processor grid, so the local atoms are the same)
    static double max_spin_diff(const std::map<tagint, std::vector<double>> &a,
                                const std::map<tagint, std::vector<double>> &b)
    {
        double diff = 0.0;
        for (const auto &s : a) {
            const auto it = b.find(s.first);
            if (it == b.end()) continue;
            for (int k = 0; k < 3; ++k)
                diff = std::max(diff, fabs(it->second[k] - s.second[k]));
        }
        double all = 0.0;
        MPI_Allreduce(&diff, &all, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        return all;
    }

    // run 200 steps with the given fix nve/spin options and check that all spins
    // moved and that their lengths and moment magnitudes are conserved
    void check_norm(const std::string &nveopt, const std::string &langevin = "")
    {
        SCOPED_TRACE("fix nve/spin " + nveopt + (langevin.empty() ? "" : ", langevin/spin " + langevin));
        build(nveopt, langevin);
        BEGIN_HIDE_OUTPUT();
        command("run 0 post no");
        END_HIDE_OUTPUT();
        const auto before = local_spins();
        double normerr, momerr;
        norm_errors(normerr, momerr);
        BEGIN_HIDE_OUTPUT();
        command("run 200 post no");
        END_HIDE_OUTPUT();
        const double change = max_spin_change(before);
        double normerr1, momerr1;
        norm_errors(normerr1, momerr1);

        EXPECT_LT(normerr, EPSILON_NORM);
        EXPECT_GT(change, 1.0e-3);    // the spins did precess
        EXPECT_LT(normerr1, EPSILON_NORM);
        EXPECT_EQ(momerr1, 0.0);

        BEGIN_HIDE_OUTPUT();
        command("clear");
        END_HIDE_OUTPUT();
    }

    // spins after 100 steps with the given fix nve/spin options
    std::map<tagint, std::vector<double>> run_spins(const std::string &nveopt)
    {
        build(nveopt);
        BEGIN_HIDE_OUTPUT();
        command("run 100 post no");
        END_HIDE_OUTPUT();
        auto spins = local_spins();
        BEGIN_HIDE_OUTPUT();
        command("clear");
        END_HIDE_OUTPUT();
        return spins;
    }
};

TEST_F(FixNVESpinTest, NormSequential)
{
    check_norm("lattice frozen");
    check_norm("lattice moving");
    check_norm("lattice moving selfconsistent 2");
    check_norm("lattice moving selfconsistent 10 tol 1e-12");
}

TEST_F(FixNVESpinTest, NormSequentialLangevin)
{
    check_norm("lattice moving", "300.0 0.1 4711");
    // the selfconsistent iterations of the sequential update only allow damping
    check_norm("lattice moving selfconsistent 2", "0.0 0.1 4711");
}

TEST_F(FixNVESpinTest, NormMidpoint)
{
    check_norm("lattice frozen integrator midpoint");
    check_norm("lattice moving integrator midpoint");
    check_norm("lattice moving integrator midpoint selfconsistent 0");
    check_norm("lattice moving integrator midpoint selfconsistent 20 tol 1e-13");
}

TEST_F(FixNVESpinTest, NormMidpointLangevin)
{
    check_norm("lattice moving integrator midpoint", "300.0 0.1 4711");
    check_norm("lattice moving integrator midpoint selfconsistent 20 tol 1e-13",
               "300.0 0.1 4711");
}

// selfconsistent 0 is the original sequential update; with tol, converged
// iterations give the same spins as a fixed, large number of iterations

TEST_F(FixNVESpinTest, SelfconsistentSettings)
{
    const auto ref  = run_spins("lattice moving");
    const auto sc0  = run_spins("lattice moving selfconsistent 0");
    const auto sc30 = run_spins("lattice moving selfconsistent 30");
    const auto tol  = run_spins("lattice moving selfconsistent 30 tol 1e-14");
    const auto mp   = run_spins("lattice moving integrator midpoint selfconsistent 30");
    const auto mpt  = run_spins("lattice moving integrator midpoint selfconsistent 30 tol 1e-14");

    EXPECT_EQ(max_spin_diff(ref, sc0), 0.0);
    EXPECT_GT(max_spin_diff(ref, sc30), 1.0e-10); // the self-dependent fields matter
    EXPECT_LT(max_spin_diff(sc30, tol), 1.0e-12);
    EXPECT_LT(max_spin_diff(mp, mpt), 1.0e-12);
}

// error messages are only printed by MPI rank 0: run this test in serial only

TEST_F(FixNVESpinTest, InvalidSettings)
{
    if (!Info::has_exceptions()) GTEST_SKIP() << "LAMMPS exceptions required";
    build_system();
    TEST_FAILURE(".*Fix nve/spin selfconsistent must be >= 0, not -1.*",
                 command("fix 1 all nve/spin selfconsistent -1"););
    TEST_FAILURE(".*Fix nve/spin tol must not be negative.*",
                 command("fix 1 all nve/spin tol -1.0"););
    TEST_FAILURE(".*Unknown fix nve/spin integrator xxx.*",
                 command("fix 1 all nve/spin integrator xxx"););
    // the iterations keyword of earlier development versions was replaced by selfconsistent
    TEST_FAILURE(".*Illegal fix/nve/spin command.*", command("fix 1 all nve/spin iterations 4"););

    // the random field would be drawn again at each iteration of a single spin update
    BEGIN_HIDE_OUTPUT();
    command("fix 1 all nve/spin selfconsistent 2");
    command("fix 3 all langevin/spin 300.0 0.1 4711");
    END_HIDE_OUTPUT();
    TEST_FAILURE(".*Fix nve/spin selfconsistent with the sequential integrator cannot be "
                 "used with fix langevin/spin 3 at a non-zero temperature.*",
                 command("run 0 post no"););

    // no such restriction for the midpoint integrator
    BEGIN_HIDE_OUTPUT();
    command("unfix 1");
    command("unfix 3");
    command("fix 1 all nve/spin integrator midpoint selfconsistent 2");
    command("fix 3 all langevin/spin 300.0 0.1 4711");
    command("run 0 post no");
    END_HIDE_OUTPUT();
}
} // namespace LAMMPS_NS

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    ::testing::InitGoogleMock(&argc, argv);

    if (LAMMPS_NS::platform::mpi_vendor() == "Open MPI" && !LAMMPS_NS::Info::has_exceptions())
        std::cout << "Warning: using OpenMPI without exceptions. "
                     "Death tests will be skipped\n";

    for (int i = 1; i < argc; ++i)
        if (strcmp(argv[i], "-v") == 0) verbose = true;

    int rv = RUN_ALL_TESTS();
    MPI_Finalize();
    return rv;
}
