import random

import torch
import numpy as np
from skimage.morphology import erosion, dilation, closing, opening, white_tophat, black_tophat

from deep_morpho.morp_operations import ParallelMorpOperations


class TestParallelMorpOperations:

    @staticmethod
    def test_init_one_layer_one_channel_input():
        """
        Test for initiating one layer with one channel input.

        Args:
        """
        morp_operation = ParallelMorpOperations(
            operations=[[[("dilation", ('disk', 3)), 'union']]]
        )
        print(morp_operation)

        morp_operation = ParallelMorpOperations(
            operations=[[[("erosion", ('hstick', 3), False), ('intersection', 0)]]]
        )
        print(morp_operation)

        morp_operation = ParallelMorpOperations(
            operations=[[[("erosion", ('hstick', 3), True), 'intersection']]]
        )
        print(morp_operation)


    @staticmethod
    def test_init_one_layer_multi_channels_input():
        """
        Test for ParallelMorpOperations that initializes one channel with multiple channels.

        Args:
        """
        ops = ["dilation", "erosion", "dilation"]
        selems = [('disk', 3), ("hstick", 7), ("vstick", 7)]
        do_comp = [True, False, False]
        aggreg = ('intersection', [0, 2])

        morp_operation = ParallelMorpOperations(
            operations=[
                [
                    [(op, selem, comp) for op, selem, comp in zip(ops, selems, do_comp)] + [aggreg]
                ]
            ]
        )
        print(morp_operation)

    @staticmethod
    def test_callable_one_layer_one_channel():
        """
        Tests that a callable that operates on one layer with one channel.

        Args:
        """
        morp_operation = ParallelMorpOperations(
            operations=[[[("erosion", ('hstick', 3), False), 'intersection']]]
        )
        morp_operation(torch.randint(0, 2, (50, 50, 1)))

        morp_operation = ParallelMorpOperations(
            operations=[[[("erosion", ('hstick', 3), True), 'intersection']]]
        )
        morp_operation(torch.randint(0, 2, (50, 50, 1)))

    @staticmethod
    def test_callable_one_layer_multi_channels():
        """
        Runs a test on a multi - channel mixture of morp operations

        Args:
        """
        for nb_chan in [2, 4, 9]:
            ops = [random.choice(['dilation', "erosion"]) for _ in range(nb_chan)]
            selems = [("disk", 3) for _ in range(nb_chan)]
            do_comp = [random.choice([True, False]) for _ in range(nb_chan)]
            aggreg = (random.choice(['intersection', 'union']), random.sample(range(nb_chan), random.choice(range(1, nb_chan+1))))
            morp_operation = ParallelMorpOperations(
                operations=[[[(op, selem, comp) for op, selem, comp in zip(ops, selems, do_comp)] + [aggreg]]]
            )

            morp_operation(torch.randint(0, 2, (50, 50, nb_chan)))


    @staticmethod
    def test_callable_multi_layers_multi_channels():
        """
        Test ParallelMorpOperations for multi layers with multi channels.

        Args:
        """

        for nb_chan in [2, 4, 9]:
            for nb_layer in [2, 4]:
                in_chan = nb_chan
                out_chans = [random.choice(range(1, 5)) for _ in range(nb_layer)]

                all_operations = []
                for layer_idx in range(nb_layer):
                    op_layer = []
                    for chan_output in range(out_chans[layer_idx]):
                        ops = [random.choice(['dilation', 'erosion']) for _ in range(in_chan)]
                        selems = [("disk", 3) for _ in range(in_chan)]
                        do_comp = [random.choice([True, False]) for _ in range(in_chan)]
                        aggreg = (random.choice(['intersection', 'union']), random.sample(range(in_chan), random.choice(range(1, in_chan+1))))
                        op_layer.append([(op, selem, comp) for op, selem, comp in zip(ops, selems, do_comp)] + [aggreg])
                    all_operations.append(op_layer)
                    in_chan = out_chans[layer_idx]

                morp_operation = ParallelMorpOperations(operations=all_operations)

                morp_operation(torch.randint(0, 2, (50, 50, nb_chan)))

    @staticmethod
    def test_erosion():
        """
        Test the erosion of a large set of images.

        Args:
        """
        for selem in [('hstick', 7), ('vstick', 7), ('disk', 3), ('scross', 7), ('dcross', 7), ('square', 7)]:
            morp_operation = ParallelMorpOperations.erosion(selem)
            inpt = np.random.randint(0, 2, (50, 50, 1))
            n = morp_operation.selems[0][0][0].shape[0] // 2 + 1
            inpt[:n] = 0
            inpt[-n:] = 0
            inpt[:, :n] = 0
            inpt[:, -n:] = 0

            ero1 = erosion(inpt[..., 0], morp_operation.selems[0][0][0])
            ero2 = morp_operation(inpt).squeeze().numpy()
            assert np.abs(ero1 - ero2).sum() == 0

    @staticmethod
    def test_dilation():
        """
        Test the dilation of a morp operation.

        Args:
        """
        for selem in [('hstick', 7), ('vstick', 7), ('disk', 3), ('scross', 7), ('dcross', 7), ('square', 7)]:
            morp_operation = ParallelMorpOperations.dilation(selem)
            inpt = np.random.randint(0, 2, (50, 50, 1))
            n = morp_operation.selems[0][0][0].shape[0] // 2 + 1
            inpt[:n] = 0
            inpt[-n:] = 0
            inpt[:, :n] = 0
            inpt[:, -n:] = 0

            ero1 = dilation(inpt[..., 0], morp_operation.selems[0][0][0])
            ero2 = morp_operation(inpt).squeeze().numpy()
            assert np.abs(ero1 - ero2).sum() == 0

    @staticmethod
    def test_opening():
        """
        Opening a random set of eros.

        Args:
        """
        for selem in [('hstick', 7), ('vstick', 7), ('disk', 3), ('scross', 7), ('dcross', 7), ('square', 7)]:
            morp_operation = ParallelMorpOperations.opening(selem)
            inpt = np.random.randint(0, 2, (50, 50, 1))
            n = morp_operation.selems[0][0][0].shape[0] // 2 + 1
            inpt[:n] = 0
            inpt[-n:] = 0
            inpt[:, :n] = 0
            inpt[:, -n:] = 0
            ero1 = opening(inpt[..., 0], morp_operation.selems[0][0][0])
            ero2 = morp_operation(inpt).squeeze().numpy()
            assert np.abs(ero1 - ero2).sum() == 0

    @staticmethod
    def test_closing():
        """
        Closeing test for ParallelMorpOperations. closing

        Args:
        """
        for selem in [('hstick', 7), ('vstick', 7), ('disk', 3), ('scross', 7), ('dcross', 7), ('square', 7)]:
            morp_operation = ParallelMorpOperations.closing(selem)
            inpt = np.random.randint(0, 2, (50, 50, 1))
            n = morp_operation.selems[0][0][0].shape[0] // 2 + 1
            inpt[:n] = 0
            inpt[-n:] = 0
            inpt[:, :n] = 0
            inpt[:, -n:] = 0

            ero1 = closing(inpt[..., 0], morp_operation.selems[0][0][0])
            ero2 = morp_operation(inpt).squeeze().numpy()
            assert np.abs(ero1 - ero2).sum() == 0

    @staticmethod
    def test_white_tophat():
        """
        Tests white tophat with different selems.

        Args:
        """
        for selem_arg in [('hstick', 7), ('vstick', 7), ('disk', 3), ('scross', 7), ('dcross', 7), ('square', 7)]:
            morp_operation = ParallelMorpOperations.white_tophat(selem_arg)
            selem = morp_operation._erodila_selem_converter(selem_arg)[0]

            inpt = np.random.randint(0, 2, (50, 50, 1))
            n = selem.shape[0] // 2 + 1
            inpt[:n] = 0
            inpt[-n:] = 0
            inpt[:, :n] = 0
            inpt[:, -n:] = 0

            ero1 = white_tophat(inpt[..., 0], selem)
            ero2 = morp_operation(inpt).squeeze().numpy()
            assert np.abs(ero1 - ero2).sum() == 0


    @staticmethod
    def test_black_tophat():
        """
        Tests black tophat with different selems.

        Args:
        """
        for selem_arg in [('hstick', 7), ('vstick', 7), ('disk', 3), ('scross', 7), ('dcross', 7), ('square', 7)]:
            morp_operation = ParallelMorpOperations.black_tophat(selem_arg)
            selem = morp_operation._erodila_selem_converter(selem_arg)[0]

            inpt = np.random.randint(0, 2, (50, 50, 1))
            n = selem.shape[0] // 2 + 1
            inpt[:n] = 0
            inpt[-n:] = 0
            inpt[:, :n] = 0
            inpt[:, -n:] = 0

            ero1 = black_tophat(inpt[..., 0], selem)
            ero2 = morp_operation(inpt).squeeze().numpy()
            assert np.abs(ero1 - ero2).sum() == 0
