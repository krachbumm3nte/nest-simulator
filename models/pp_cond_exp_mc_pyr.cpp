/*
 *  pp_cond_exp_mc_pyr.cpp
 *
 *  This file is part of NEST.
 *
 *  Copyright (C) 2004 The NEST Initiative
 *
 *  NEST is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  NEST is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with NEST.  If not, see <http://www.gnu.org/licenses/>.
 *
 */


#include "pp_cond_exp_mc_pyr.h"

#ifdef HAVE_GSL

// C++ includes:
#include <cstdio>
#include <iostream>

// Includes from libnestutil:
#include "numerics.h"

// Includes from nestkernel:
#include "exceptions.h"
#include "kernel_manager.h"
#include "universal_data_logger_impl.h"

// Includes from sli:
#include "dict.h"
#include "dictutils.h"

/* ----------------------------------------------------------------
 * Compartment name list
 * ---------------------------------------------------------------- */

/* Harold Gutch reported some static destruction problems on OSX 10.4.
   He pointed out that the problem is avoided by defining the comp_names_
   vector with its final size. See also #348.
*/
std::vector< Name > nest::pp_cond_exp_mc_pyr::comp_names_( NCOMP );

/* ----------------------------------------------------------------
 * Receptor dictionary
 * ---------------------------------------------------------------- */

// leads to seg fault on exit, see #328
// DictionaryDatum nest::pp_cond_exp_mc_pyr::receptor_dict_ = new
// Dictionary();

/* ----------------------------------------------------------------
 * Recordables map
 * ---------------------------------------------------------------- */

nest::RecordablesMap< nest::pp_cond_exp_mc_pyr > nest::pp_cond_exp_mc_pyr::recordablesMap_;

namespace nest
{
// specialization must be place in namespace

template <>
void
RecordablesMap< pp_cond_exp_mc_pyr >::create()
{
  insert_(
    Name( "V_m.s" ), &pp_cond_exp_mc_pyr::get_y_elem_< pp_cond_exp_mc_pyr::State_::V_M, pp_cond_exp_mc_pyr::SOMA > );
  insert_(
    Name( "V_m.b" ), &pp_cond_exp_mc_pyr::get_y_elem_< pp_cond_exp_mc_pyr::State_::V_M, pp_cond_exp_mc_pyr::BASAL > );
  insert_( Name( "V_m.a_lat" ),
    &pp_cond_exp_mc_pyr::get_y_elem_< pp_cond_exp_mc_pyr::State_::V_M, pp_cond_exp_mc_pyr::APICAL_LAT > );
  // insert_( Name( "V_m.a_td" ),
  //   &pp_cond_exp_mc_pyr::get_y_elem_< pp_cond_exp_mc_pyr::State_::V_M, pp_cond_exp_mc_pyr::APICAL_TD > );
}
}

/* ----------------------------------------------------------------
 * Iteration function
 * ---------------------------------------------------------------- */

extern "C" int
nest::pp_cond_exp_mc_pyr_dynamics( double, const double y[], double f[], void* pnode )
{
  // some shorthands
  typedef nest::pp_cond_exp_mc_pyr N;
  typedef nest::pp_cond_exp_mc_pyr::State_ S;

  // get access to node so we can work almost as in a member function
  assert( pnode );
  const nest::pp_cond_exp_mc_pyr& node = *( reinterpret_cast< nest::pp_cond_exp_mc_pyr* >( pnode ) );

  // computations written quite explicitly for clarity, assume compile
  // will optimized most stuff away ... TODO: understand the extent of this

  // membrane potential of soma
  const double V = y[ S::idx( N::SOMA, S::V_M ) ];

  // leak current of soma
  const double I_L = node.P_.pyr_params.g_L[ N::SOMA ] * V;

  // coupling from dendrites to soma all summed up
  double I_conn_d_s = 0.0;

  // compute dynamics for each dendritic compartment
  // computations written quite explicitly for clarity, assume compile
  // will optimized most stuff away ...
  for ( size_t n = 1; n < N::NCOMP; ++n )
  {
    // membrane potential of dendrite
    const double V_dnd = y[ S::idx( n, S::V_M ) ];

    // coupling current from dendrite to soma
    I_conn_d_s += node.P_.pyr_params.g_conn[ n ] * V_dnd;

    // dendritic current due to input
    const double I_dend = y[ S::idx( n, S::I ) ];

    const double I_L_dend = -node.P_.pyr_params.g_L[ n ] * V_dnd;

    // derivative membrane potential
    f[ S::idx( n, S::V_M ) ] = ( I_L_dend + I_dend * 0.1 );

    // derivative dendritic current
    f[ S::idx( n, S::I ) ] = -I_dend / node.P_.pyr_params.tau_m;
    if ( n == 1 )
    {
      std::cout << "foo: " << I_L_dend << ", " << I_conn_d_s << ", " << y[ S::idx( 1, S::V_M ) ] << ", "
                << y[ S::idx( 0, S::V_M ) ] << std::endl;
    }
  }

  // derivative membrane potential
  // soma
  f[ S::idx( N::SOMA, S::V_M ) ] = -I_L + I_conn_d_s + node.B_.I_stim_[ N::SOMA ] + node.P_.I_e[ N::SOMA ];

  // excitatory conductance soma
  f[ S::idx( N::SOMA, S::I ) ] = -y[ S::idx( N::SOMA, S::I ) ] / node.P_.pyr_params.tau_m;
  // std::cout << "soma curr " << y[ S::idx( N::SOMA, S::I)] << "\n";

  return GSL_SUCCESS;
}

/* ----------------------------------------------------------------
 * Default constructors defining default parameters and state
 * ---------------------------------------------------------------- */

nest::pp_cond_exp_mc_pyr::Parameters_::Parameters_()
  : t_ref( 3.0 ) // ms
{
  // parameters for the transfer function
  pyr_params.phi_max = 1;
  pyr_params.gamma = 1;
  pyr_params.beta = 1;
  pyr_params.theta = 0;

  // conductances between compartments
  pyr_params.tau_m = 1.0;


  pyr_params.curr_target = 0;
  pyr_params.lambda_curr = 0.0;
  pyr_params.C_m = 1.0; // pF
  pyr_params.use_phi = true;


  // soma parameters
  pyr_params.g_conn[ SOMA ] = 0.0; // nS, soma-dendrite
  pyr_params.g_L[ SOMA ] = 1.9;    // nS
  pyr_params.E_L[ SOMA ] = 0.0;    // mV
  I_e[ SOMA ] = 0.0;               // pA

  // basal dendrite parameters
  pyr_params.g_conn[ BASAL ] = 1.0; // nS, dendrite-soma
  pyr_params.g_L[ BASAL ] = 0.0;
  pyr_params.E_L[ BASAL ] = 0.0; // mV
  I_e[ BASAL ] = 0.0;            // pA

  // proximal apical dendrite parameters
  pyr_params.g_conn[ APICAL_LAT ] = 0.8;
  pyr_params.g_L[ APICAL_LAT ] = 0.0;
  pyr_params.E_L[ APICAL_LAT ] = 0.0; // mV
  I_e[ APICAL_LAT ] = 0.0;            // pA

  // distal apical dendrite parameters
  // pyr_params.g_conn[ APICAL_TD ] = 0.8;
  // pyr_params.g_L[ APICAL_TD ] = 0.0;
  // pyr_params.E_L[ APICAL_TD ] = 0.0; // mV
  // I_e[ APICAL_TD ] = 0.0;            // pA
}

nest::pp_cond_exp_mc_pyr::Parameters_::Parameters_( const Parameters_& p )
  : t_ref( p.t_ref )
{
  pyr_params.phi_max = p.pyr_params.phi_max;
  pyr_params.gamma = p.pyr_params.gamma;
  pyr_params.beta = p.pyr_params.beta;
  pyr_params.theta = p.pyr_params.theta;

  pyr_params.curr_target = p.pyr_params.curr_target;
  pyr_params.lambda_curr = p.pyr_params.lambda_curr;
  pyr_params.tau_m = p.pyr_params.tau_m;
  pyr_params.C_m = p.pyr_params.C_m;
  pyr_params.use_phi = p.pyr_params.use_phi;

  // copy C-arrays

  for ( size_t n = 0; n < NCOMP; ++n )
  {
    pyr_params.g_conn[ n ] = p.pyr_params.g_conn[ n ];
    pyr_params.g_L[ n ] = p.pyr_params.g_L[ n ];
    pyr_params.E_L[ n ] = p.pyr_params.E_L[ n ];
    I_e[ n ] = p.I_e[ n ];
  }
}

nest::pp_cond_exp_mc_pyr::Parameters_&
nest::pp_cond_exp_mc_pyr::Parameters_::operator=( const Parameters_& p )
{
  assert( this != &p ); // would be bad logical error in program

  t_ref = p.t_ref;
  pyr_params.phi_max = p.pyr_params.phi_max;
  pyr_params.gamma = p.pyr_params.gamma;
  pyr_params.beta = p.pyr_params.beta;
  pyr_params.theta = p.pyr_params.theta;
  pyr_params.tau_m = p.pyr_params.tau_m;

  pyr_params.curr_target = p.pyr_params.curr_target;
  pyr_params.lambda_curr = p.pyr_params.lambda_curr;
  pyr_params.C_m = p.pyr_params.C_m;
  pyr_params.use_phi = p.pyr_params.use_phi;

  for ( size_t n = 0; n < NCOMP; ++n )
  {
    pyr_params.g_conn[ n ] = p.pyr_params.g_conn[ n ];
    pyr_params.g_L[ n ] = p.pyr_params.g_L[ n ];
    pyr_params.E_L[ n ] = p.pyr_params.E_L[ n ];
    I_e[ n ] = p.I_e[ n ];
  }

  return *this;
}


nest::pp_cond_exp_mc_pyr::State_::State_( const Parameters_& p )
  : r_( 0 )
{
  // for simplicity, we first initialize all values to 0,
  // then set the membrane potentials for each compartment
  for ( size_t i = 0; i < STATE_VEC_SIZE; ++i )
  {
    y_[ i ] = 0;
  }
  for ( size_t n = 0; n < NCOMP; ++n )
  {
    y_[ idx( n, V_M ) ] = p.pyr_params.E_L[ n ];
  }
}

nest::pp_cond_exp_mc_pyr::State_::State_( const State_& s )
  : r_( s.r_ )
{
  for ( size_t i = 0; i < STATE_VEC_SIZE; ++i )
  {
    y_[ i ] = s.y_[ i ];
  }
}

nest::pp_cond_exp_mc_pyr::State_&
nest::pp_cond_exp_mc_pyr::State_::operator=( const State_& s )
{
  r_ = s.r_;
  for ( size_t i = 0; i < STATE_VEC_SIZE; ++i )
  {
    y_[ i ] = s.y_[ i ];
  }
  return *this;
}

nest::pp_cond_exp_mc_pyr::Buffers_::Buffers_( pp_cond_exp_mc_pyr& n )
  : logger_( n )
  , s_( nullptr )
  , c_( nullptr )
  , e_( nullptr )
{
  // Initialization of the remaining members is deferred to
  // init_buffers_().
}

nest::pp_cond_exp_mc_pyr::Buffers_::Buffers_( const Buffers_&, pp_cond_exp_mc_pyr& n )
  : logger_( n )
  , s_( nullptr )
  , c_( nullptr )
  , e_( nullptr )
{
  // Initialization of the remaining members is deferred to
  // init_buffers_().
}

/* ----------------------------------------------------------------
 * Parameter and state extractions and manipulation functions
 * ---------------------------------------------------------------- */

void
nest::pp_cond_exp_mc_pyr::Parameters_::get( DictionaryDatum& d ) const
{
  def< double >( d, names::t_ref, t_ref );
  def< double >( d, names::phi_max, pyr_params.phi_max );
  def< double >( d, names::gamma, pyr_params.gamma );
  def< double >( d, names::beta, pyr_params.beta );
  def< double >( d, names::theta, pyr_params.theta );
  def< double >( d, names::tau_m, pyr_params.tau_m );


  def< double >( d, names::g_som, pyr_params.g_conn[ SOMA ] );
  def< double >( d, names::g_b, pyr_params.g_conn[ BASAL ] );
  def< double >( d, names::g_a, pyr_params.g_conn[ APICAL_LAT ] );

  def< double >( d, names::lambda, pyr_params.lambda_curr );
  def< double >( d, names::target, pyr_params.curr_target );


  // create subdictionaries for per-compartment parameters
  for ( size_t n = 0; n < NCOMP; ++n )
  {
    DictionaryDatum dd = new Dictionary();

    def< double >( dd, names::g, pyr_params.g_conn[ n ] );
    def< double >( dd, names::g_L, pyr_params.g_L[ n ] );
    def< double >( dd, names::E_L, pyr_params.E_L[ n ] );
    def< double >( dd, names::I_e, I_e[ n ] );

    ( *d )[ comp_names_[ n ] ] = dd;
  }
}

void
nest::pp_cond_exp_mc_pyr::Parameters_::set( const DictionaryDatum& d )
{
  // allow setting the membrane potential
  updateValue< double >( d, names::t_ref, t_ref );
  updateValue< double >( d, names::phi_max, pyr_params.phi_max );
  updateValue< double >( d, names::gamma, pyr_params.gamma );
  updateValue< double >( d, names::beta, pyr_params.beta );
  updateValue< double >( d, names::theta, pyr_params.theta );
  updateValue< double >( d, names::tau_m, pyr_params.tau_m );
  updateValue< double >( d, names::C_m, pyr_params.C_m );


  updateValue< double >( d, Name( names::g_som ), pyr_params.g_conn[ SOMA ] );
  updateValue< double >( d, Name( names::g_b ), pyr_params.g_conn[ BASAL ] );
  updateValue< double >( d, Name( names::g_a ), pyr_params.g_conn[ APICAL_LAT ] );

  updateValue< double >( d, Name( names::target ), pyr_params.curr_target );
  updateValue< double >( d, Name( names::lambda ), pyr_params.lambda_curr );
  updateValue< bool >( d, Name( names::use_phi ), pyr_params.use_phi );


  // extract from sub-dictionaries
  for ( size_t n = 0; n < NCOMP; ++n )
  {
    if ( d->known( comp_names_[ n ] ) )
    {
      DictionaryDatum dd = getValue< DictionaryDatum >( d, comp_names_[ n ] );

      updateValue< double >( dd, names::E_L, pyr_params.E_L[ n ] );
      updateValue< double >( dd, names::g, pyr_params.g_conn[ n ] );
      updateValue< double >( dd, names::g_L, pyr_params.g_L[ n ] );
      updateValue< double >( dd, names::I_e, I_e[ n ] );
    }
  }
  if ( pyr_params.gamma < 0 )
  {
    throw BadProperty( "Rate slope cannot be negative." );
  }

  if ( pyr_params.phi_max < 0 )
  {
    throw BadProperty( "Maximum rate cannot be negative." );
  }

  if ( t_ref < 0 )
  {
    throw BadProperty( "Refractory time cannot be negative." );
  }

  if ( pyr_params.C_m <= 0 )
  {
    throw BadProperty( "Capacitance must be strictly positive." );
  }

  if ( pyr_params.tau_m <= 0 )
  {
    throw BadProperty( "Synaptic time constant must be strictly positive." );
  }
}

void
nest::pp_cond_exp_mc_pyr::State_::get( DictionaryDatum& d ) const
{
  // we assume here that State_::get() always is called after
  // Parameters_::get(), so that the per-compartment dictionaries exist
  for ( size_t n = 0; n < NCOMP; ++n )
  {
    assert( d->known( comp_names_[ n ] ) );
    DictionaryDatum dd = getValue< DictionaryDatum >( d, comp_names_[ n ] );

    def< double >( dd, names::V_m, y_[ idx( n, V_M ) ] ); // Membrane potential
  }
}

void
nest::pp_cond_exp_mc_pyr::State_::set( const DictionaryDatum& d, const Parameters_& )
{
  // extract from sub-dictionaries
  for ( size_t n = 0; n < NCOMP; ++n )
  {
    if ( d->known( comp_names_[ n ] ) )
    {
      DictionaryDatum dd = getValue< DictionaryDatum >( d, comp_names_[ n ] );
      updateValue< double >( dd, names::V_m, y_[ idx( n, V_M ) ] );
    }
  }
}


/* ----------------------------------------------------------------
 * Default and copy constructor for node, and destructor
 * ---------------------------------------------------------------- */

nest::pp_cond_exp_mc_pyr::pp_cond_exp_mc_pyr()
  : PyrArchivingNode< pp_cond_exp_mc_pyr_parameters >()
  , P_()
  , S_( P_ )
  , B_( *this )
{
  recordablesMap_.create();

  // set up table of compartment names
  // comp_names_.resize(NCOMP); --- Fixed size, see comment on definition
  comp_names_[ SOMA ] = Name( "soma" );
  comp_names_[ BASAL ] = Name( "basal" );
  comp_names_[ APICAL_LAT ] = Name( "apical_lat" );
  // comp_names_[ APICAL_TD ] = Name( "apical_td" );
  PyrArchivingNode< pp_cond_exp_mc_pyr_parameters >::pyr_params = &P_.pyr_params;
}

nest::pp_cond_exp_mc_pyr::pp_cond_exp_mc_pyr( const pp_cond_exp_mc_pyr& n )
  : PyrArchivingNode< pp_cond_exp_mc_pyr_parameters >( n )
  , P_( n.P_ )
  , S_( n.S_ )
  , B_( n.B_, *this )
{
  PyrArchivingNode< pp_cond_exp_mc_pyr_parameters >::pyr_params = &P_.pyr_params;
}

nest::pp_cond_exp_mc_pyr::~pp_cond_exp_mc_pyr()
{
  // GSL structs may not have been allocated, so we need to protect destruction
  if ( B_.s_ )
  {
    gsl_odeiv_step_free( B_.s_ );
  }
  if ( B_.c_ )
  {
    gsl_odeiv_control_free( B_.c_ );
  }
  if ( B_.e_ )
  {
    gsl_odeiv_evolve_free( B_.e_ );
  }
}

/* ----------------------------------------------------------------
 * Node initialization functions
 * ---------------------------------------------------------------- */

void
nest::pp_cond_exp_mc_pyr::init_buffers_()
{
  B_.spikes_.resize( NUM_SPIKE_RECEPTORS );
  for ( size_t n = 0; n < NUM_SPIKE_RECEPTORS; ++n )
  {
    B_.spikes_[ n ].clear();
  } // includes resize

  B_.currents_.resize( NUM_CURR_RECEPTORS );
  for ( size_t n = 0; n < NUM_CURR_RECEPTORS; ++n )
  {
    B_.currents_[ n ].clear(); // includes resize
  }

  B_.logger_.reset();
  ArchivingNode::clear_history();

  B_.step_ = Time::get_resolution().get_ms();
  B_.IntegrationStep_ = B_.step_;

  if ( not B_.s_ )
  {
    B_.s_ = gsl_odeiv_step_alloc( gsl_odeiv_step_rkf45, State_::STATE_VEC_SIZE );
  }
  else
  {
    gsl_odeiv_step_reset( B_.s_ );
  }

  if ( not B_.c_ )
  {
    B_.c_ = gsl_odeiv_control_y_new( 1e-3, 0.0 );
  }
  else
  {
    gsl_odeiv_control_init( B_.c_, 1e-3, 0.0, 1.0, 0.0 );
  }

  if ( not B_.e_ )
  {
    B_.e_ = gsl_odeiv_evolve_alloc( State_::STATE_VEC_SIZE );
  }
  else
  {
    gsl_odeiv_evolve_reset( B_.e_ );
  }

  B_.sys_.function = pp_cond_exp_mc_pyr_dynamics;
  B_.sys_.jacobian = nullptr;
  B_.sys_.dimension = State_::STATE_VEC_SIZE;
  B_.sys_.params = reinterpret_cast< void* >( this );
  for ( size_t n = 0; n < NCOMP; ++n )
  {
    B_.I_stim_[ n ] = 0.0;
  }
}

void
nest::pp_cond_exp_mc_pyr::pre_run_hook()
{
  // ensures initialization in case mm connected after Simulate
  B_.logger_.init();
  V_.rng_ = get_vp_specific_rng( get_thread() );

  V_.RefractoryCounts_ = Time( Time::ms( P_.t_ref ) ).get_steps();

  V_.h_ = Time::get_resolution().get_ms();
  // since t_ref >= 0, this can only fail in error
  assert( V_.RefractoryCounts_ >= 0 );

  assert( ( int ) NCOMP == ( int ) pp_cond_exp_mc_pyr_parameters::NCOMP );
}


/* ----------------------------------------------------------------
 * Update and spike handling functions
 * ---------------------------------------------------------------- */

void
nest::pp_cond_exp_mc_pyr::update( Time const& origin, const long from, const long to )
{

  assert( to >= 0 and ( delay ) from < kernel().connection_manager.get_min_delay() );
  assert( from < to );

  for ( long lag = from; lag < to; ++lag )
  {
    /*

    double t = 0.0;

    // numerical integration with adaptive step size control:
    // ------------------------------------------------------
    // gsl_odeiv_evolve_apply performs only a single numerical
    // integration step, starting from t and bounded by step;
    // the while-loop ensures integration over the whole simulation
    // step (0, step] if more than one integration step is needed due
    // to a small integration step size;
    // note that (t+IntegrationStep > step) leads to integration over
    // (t, step] and afterwards setting t to step, but it does not
    // enforce setting IntegrationStep to step-t; this is of advantage
    // for a consistent and efficient integration across subsequent
    // simulation intervals
    while ( t < B_.step_ )
    {
      const int status = gsl_odeiv_evolve_apply( B_.e_,
        B_.c_,
        B_.s_,
        &B_.sys_,             // system of ODE
        &t,                   // from t
        B_.step_,             // to t <= step
        &B_.IntegrationStep_, // integration step size
        S_.y_ );              // neuronal state
      if ( status != GSL_SUCCESS )
      {
        throw GSLSolverFailure( get_name(), status );
      }
    }
    // std::cout  <<  S_.y_[ State_::idx( 0, State_::V_M ) ] << ", " << P_.I_e[0] << std::endl;
    */


    // membrane potential of soma
    const double V = S_.y_[ State_::idx( P_.pyr_params.SOMA, State_::V_M ) ];

    // leak current of soma
    // TODO: I am misappropriating g_som here, because I am too lazy to create another neuron parameter.
    // The gist of it is, that all neuron types have effective leakage conductance of (g_l + g_D + g_A),
    // yet in- and output neurons have an apical conductance of zero. Thus we need to store this sum separately.
    // as it serves no other purpose, the somatic conductance is used here (temporarily?)
    const double I_L = P_.pyr_params.g_conn[ pyr_params->SOMA ] * V;

    // coupling from dendrites to soma all summed up
    double I_conn_d_s = 0.0;

    // compute dynamics for each dendritic compartment
    // computations written quite explicitly for clarity, assume compile
    // will optimized most stuff away ...
    for ( size_t n = 1; n < P_.pyr_params.NCOMP; ++n )
    {
      // membrane potential of dendrite
      const double V_dnd = S_.y_[ State_::idx( n, State_::V_M ) ];

      // coupling current from dendrite to soma
      I_conn_d_s += P_.pyr_params.g_conn[ n ] * V_dnd;

      // dendritic current due to input
      const double I_dend = S_.y_[ State_::idx( n, State_::I ) ];

      const double I_L_dend = P_.pyr_params.g_L[ n ] * V_dnd;

      // derivative membrane potential
      S_.y_[ State_::idx( n, State_::V_M ) ] = V_dnd + ( -I_L_dend + I_dend );

      // derivative dendritic current
      S_.y_[ State_::idx( n, State_::I ) ] = I_dend - I_dend / P_.pyr_params.tau_m;
      // if ( n == 10 )
      //{
      //   std::cout << I_L_dend << ", " << I_conn_d_s << ", " << S_.y_[ State_::idx( 1, State_::V_M ) ] << ", "
      //             << S_.y_[ State_::idx( 0, State_::V_M ) ] << std::endl;
      // }
    }

    // derivative membrane potential
    // soma
    S_.y_[ State_::idx( P_.pyr_params.SOMA, State_::V_M ) ] =
      V + 0.1 * ( -I_L + I_conn_d_s + B_.I_stim_[ P_.pyr_params.SOMA ] + P_.I_e[ P_.pyr_params.SOMA ] );

    // excitatory conductance soma
    S_.y_[ State_::idx( P_.pyr_params.SOMA, State_::I ) ] =
      0; //-S_.y_[ State_::idx( P_.pyr_params.SOMA, State_::I ) ] / P_.pyr_params.tau_m;
         // std::cout << "soma curr " << y[ S::idx( N::SOMA, S::I)] << "\n";


    // add incoming spikes to all compartmens
    for ( size_t n = 0; n < NCOMP; ++n )
    {
      S_.y_[ State_::idx( n, State_::I ) ] += B_.spikes_[ n ].get_value( lag );
    }


    SpikeEvent se;
    se.set_sender( *this );
    kernel().event_delivery_manager.send( *this, se, lag );


    for ( size_t n = 0; n < NCOMP; ++n )
    {
      B_.I_stim_[ n ] = 0.0;
    }


    /*
    // Declaration outside if statement because we need it later
    unsigned long n_spikes = 0;

    if ( S_.r_ == 0 )
    {
      // Neuron not refractory

      // There is no reset of the membrane potential after a spike
      double rate = P_.pyr_params.phi( S_.y_[ State_::V_M ] );

      if ( rate > 0.0 )
      {

        if ( P_.t_ref > 0.0 )
        {
          // Draw random number and compare to prob to have a spike
          if ( V_.rng_->drand() <= -numerics::expm1( -rate * V_.h_ ) )
          {
            n_spikes = 1;
          }
        }
        else
        {
          // Draw Poisson random number of spikes
          poisson_distribution::param_type param( rate * V_.h_ );
          n_spikes = V_.poisson_dist_( V_.rng_, param );
        }

        if ( n_spikes > 0 ) // Is there a spike? Then set the new dead time.
        {
          // Set dead time interval according to parameters
          S_.r_ = V_.RefractoryCounts_;

          // And send the spike event
          SpikeEvent se;
          se.set_multiplicity( n_spikes );
          kernel().event_delivery_manager.send( *this, se, lag );

          // Set spike time in order to make plasticity rules work
          for ( unsigned int i = 0; i < n_spikes; i++ )
          {
            set_spiketime( Time::step( origin.get_steps() + lag + 1 ) );
          }
        }
      } // if (rate > 0.0)
    }
    else // Neuron is within dead time
    {
      --S_.r_;
    }*/

    /*
    // Store dendritic membrane potential for Urbanczik-Senn plasticity
    write_urbanczik_history( Time::step( origin.get_steps() + lag + 1 ),
      S_.y_[ S_.idx( BASAL, State_::V_M ) ],
      S_.y_[ S_.idx( SOMA, State_::V_M ) ],
      BASAL );

    write_urbanczik_history( Time::step( origin.get_steps() + lag + 1 ),
      S_.y_[ S_.idx( APICAL_TD, State_::V_M ) ],
      S_.y_[ S_.idx( SOMA, State_::V_M ) ],
      APICAL_TD );
    write_urbanczik_history( Time::step( origin.get_steps() + lag + 1 ),
      S_.y_[ S_.idx( APICAL_LAT, State_::V_M ) ],
      S_.y_[ S_.idx( SOMA, State_::V_M ) ],
      APICAL_LAT );
    */

    // set new input currents
    for ( size_t n = 0; n < NCOMP; ++n )
    {
      B_.I_stim_[ n ] = B_.currents_[ n ].get_value( lag );
    }

    // log state data
    B_.logger_.record_data( origin.get_steps() + lag );


    // send current event to target

    if ( P_.pyr_params.curr_target != 0 )
    {
      CurrentEvent ce;
      ce.set_current( S_.y_[ S_.idx( SOMA, State_::V_M ) ] );
      Node* n = kernel().node_manager.get_node_or_proxy( P_.pyr_params.curr_target );
      ce.set_receiver( *n );
      ce.set_sender_node_id( this->get_node_id() );
      ce.set_rport( 0 ); // TODO: make this flexible to target not only the soma!
      ce.set_sender( *this );
      ce.set_stamp( Time::step( origin.get_steps() + lag + 1 ) );
      ce.set_weight( P_.pyr_params.lambda_curr );
      ce();
    }
  }
}

void
nest::pp_cond_exp_mc_pyr::handle( SpikeEvent& e )
{
  assert( e.get_delay_steps() > 0 );
  assert( 0 <= e.get_rport() and e.get_rport() < 2 * NCOMP );

  // double spike_val = e.get_weight() * e.get_multiplicity();
  //  We multiply with the sender activity here because it allows us to track weights with a weight_recorder.

  Node* sender = kernel().node_manager.get_node_or_proxy( e.retrieve_sender_node_id_from_source_table() );
  nest::pp_cond_exp_mc_pyr* sender_pyr = static_cast< nest::pp_cond_exp_mc_pyr* >( sender );

  double spike_val = e.get_weight() * sender_pyr->pyr_params->phi( sender_pyr->get_V_m( 0 ) );


  B_.spikes_[ e.get_rport() ].add_value(
    e.get_rel_delivery_steps( kernel().simulation_manager.get_slice_origin() ), spike_val );
}


void
nest::pp_cond_exp_mc_pyr::handle( CurrentEvent& e )
{
  assert( e.get_delay_steps() > 0 );
  // TODO: not 100% clean, should look at MIN, SUP
  assert( 0 <= e.get_rport() and e.get_rport() < NCOMP );

  // add weighted current; HEP 2002-10-04
  B_.currents_[ e.get_rport() ].add_value(
    e.get_rel_delivery_steps( kernel().simulation_manager.get_slice_origin() ), e.get_weight() * e.get_current() );
}

void
nest::pp_cond_exp_mc_pyr::handle( DataLoggingRequest& e )
{
  B_.logger_.handle( e );
}

#endif // HAVE_GSL