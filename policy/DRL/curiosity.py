


def curiosity_backprop(self, prev_state, state, action, mpc):
    from constants_prediction_curiosity import constants
    predictor = mpc.StateActionPredictor(len(prev_state), len(action), designHead='pydial')

    # # computing predictor loss
    # if self.unsup:
    #     if 'state' in unsupType:
    #         self.predloss = constants['PREDICTION_LR_SCALE'] * predictor.forwardloss
    #     else:
    self.predloss = constants['PREDICTION_LR_SCALE'] * (predictor.invloss * (1 - constants['FORWARD_LOSS_WT']) +
                                                        predictor.forwardloss * constants['FORWARD_LOSS_WT'])
    predgrads = tf.gradients(self.predloss * 20.0,  # our batch size is?
                             predictor.var_list)  # batchsize=20. Factored out to make hyperparams not depend on it.

    predgrads, _ = tf.clip_by_global_norm(predgrads, constants['GRAD_NORM_CLIP'])
    pred_grads_and_vars = list(zip(predgrads, predictor.var_list))
    grads_and_vars = pred_grads_and_vars  # prediction only here for now, do i want to combine it with policy?
    # each worker has a different set of adam optimizer parameters
    # make optimizer global shared, if needed
    print("Optimizer: ADAM with lr: %f" % (constants['LEARNING_RATE']))
    # print("Input observation shape: ", env.observation_space.shape)
    opt = tf.train.AdamOptimizer(constants['LEARNING_RATE'])
    # train_op = tf.group(opt.apply_gradients(grads_and_vars), inc_step)
    with tf.variable_scope('what_goes_here', reuse=tf.AUTO_REUSE):  # why? and what goes into varscope?
        train_op = opt.apply_gradients(grads_and_vars)
        # next, run op session
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        feed_dict = {predictor.s1: [prev_state],
                     predictor.s2: [state],
                     predictor.asample: [action]
                     }
        sess.run(train_op, feed_dict=feed_dict)
    return train_op, self.predloss