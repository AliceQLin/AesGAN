import tensorflow as tf
from utils import mkdir_p, Eyes
from AesGAN import AesGAN
import os

tf.reset_default_graph()

flags = tf.app.flags
flags.DEFINE_integer("OPER_FLAG", 0, "flag of opertion, test or train")
flags.DEFINE_string("OPER_NAME", "Experiment", "name of the experiment")
flags.DEFINE_string("path", './celeb_id_aligned/', "path of training data")
flags.DEFINE_integer("batch_size", 4, "size of single batch")
flags.DEFINE_integer("max_iters", 100000, "number of total iterations for G")
flags.DEFINE_float("learn_rate", 0.0001, "learning rate for g and d")
flags.DEFINE_integer("test_step", 100000, "loading setp model for testing")
flags.DEFINE_boolean("is_load", False, "whether loading the pretraining model for training")
flags.DEFINE_boolean("use_sp", True, "whether using spectral normalization")
flags.DEFINE_integer("lam_recon", 1, "weight for recon loss")
flags.DEFINE_integer("lam_gp", 10, "weight for gradient penalty")
flags.DEFINE_float("beta1", 0.5, "beta1 of Adam optimizer")
flags.DEFINE_float("beta2", 0.999, "beta2 of Adam optimizer")
flags.DEFINE_integer("n_critic", 1, "iters of g for every d")

FLAGS = flags.FLAGS

if __name__ == "__main__":

    print(FLAGS.OPER_FLAG)
    
    
    root_log_dir = "./output/log/logs{}".format(FLAGS.OPER_FLAG)
    checkpoint_dir = "./output/model_gan{}/".format(FLAGS.OPER_NAME)
    sample_path = "./output/sample{}/sample_{}".format(FLAGS.OPER_FLAG, FLAGS.OPER_NAME)
   
    mkdir_p(root_log_dir)
    mkdir_p(checkpoint_dir)
    mkdir_p(sample_path)

    m_ob = Eyes(FLAGS.path)

    eGan = AesGAN(batch_size= FLAGS.batch_size, max_iters= FLAGS.max_iters,
                  model_path= checkpoint_dir, data_ob= m_ob, sample_path= sample_path , log_dir= root_log_dir,
                  learning_rate=  FLAGS.learn_rate, is_load=FLAGS.is_load, lam_recon=FLAGS.lam_recon, lam_gp=FLAGS.lam_gp,
                  use_sp=FLAGS.use_sp, beta1=FLAGS.beta1, beta2=FLAGS.beta2, n_critic=FLAGS.n_critic)

    if FLAGS.OPER_FLAG == 0:
        eGan.build_model_GAN()
        eGan.train()

    if FLAGS.OPER_FLAG == 1:
        eGan.build_test_model_GAN()
        eGan.test(test_step=FLAGS.test_step)


