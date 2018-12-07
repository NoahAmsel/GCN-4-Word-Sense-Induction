#!/usr/bin/perl

#@@General description

use strict;
use IO::File;
use File::Basename;
use Getopt::Std;
use Data::Dumper;

my %opts;

getopt('prO', \%opts);

my $opt_p = $opts{'p'} ? $opts{'p'} : 'all';

{
  &usage("Too few input files.") unless (scalar (@ARGV) == 2);
  &usage("undefined $opt_p option for -p switch.") unless ($opt_p =~ /^(all|n|v)/);

  my $file_in = shift @ARGV;
  my $train_key_file = shift @ARGV;

  my $map_out_dir = $opts{'O'} ? $opts{'O'} : 0;
  $map_out_dir =~ s/\/$//; # skip trailing slash

  &usage("Can't open key file $train_key_file") unless (-f $train_key_file);
  &usage("Can't open file $file_in") unless (-f $file_in);

  &main($file_in, $train_key_file, $map_out_dir);
}

sub main {

  my ($file_in, $train_key_file, $map_out_dir) = @_;

  my %CSolution;  # Clustering solution
  my %CSol_gs;    # Clustering solution Gold Standard (train)

  &read_cluster_solution($file_in, \%CSolution);
  &read_gold_standard($train_key_file, \%CSol_gs);

  my %Sense_solution;

  foreach my $word (keys %CSolution) {

    next if &discard_word($word, $opt_p);

    my @c2s;

    my %sense_freq = &get_gs_senses_freq($CSol_gs{$word});    # { sense => frequency } (for mfs)

    my @clusters = &get_clusters($CSolution{$word});
    my @senses = sort keys %sense_freq;

    my %cluster2idx = &hash_idx(\@clusters);
    my %sense2idx = &hash_idx(\@senses);

    my %SenseSv;
    my @train_instV = keys(%{$CSol_gs{$word}}); # train instances
    &senses_score_vector(\@train_instV, $CSolution{$word}, $CSol_gs{$word}, \%cluster2idx, \%SenseSv);

    my $M = &create_mapping_matrix(\%SenseSv, \%sense2idx, \%cluster2idx);

    &write_cluster2sense($map_out_dir, $word, \%sense_freq, \@senses, \@clusters, $M) if ($map_out_dir);

    # Now, label the test instances with senses (create a "sense solution")

    $Sense_solution{$word} = {};
    my @test_instV = &get_test_instances($CSolution{$word}, $CSol_gs{$word});
    &label_instances_sense(\@test_instV, $CSolution{$word}, \%cluster2idx, \@senses, $M, $Sense_solution{$word});
  }
  &write_sense_solution(\%Sense_solution);
}



sub write_sense_solution {

  my ($ssol) = @_;

  my @Sol;

  foreach my $word (keys %{$ssol}) {
    foreach my $inst (keys %{$ssol->{$word}}) {
      push (@Sol, "$word $inst $ssol->{$word}->{$inst}");
    }
  }
  print join("\n", sort @Sol);
  print "\n";
}

# Label the instances with appropiated senses
#
# 3 steps:
#
# 1. get cluster_score_vector of instance (cluster_sv)
# 2. multiply cluster_sv*M and get a sense_score_vector (sense_sv)
# 3. label instance with the sense of maximum score if sense_sv has an
#    element > 0


sub label_instances_sense {

  my ($instV, $csol, $c2idx, $senses, $M, $ssol) = @_;

  foreach my $inst (@{$instV}) {
    my @cluster_sv = &instance_cluster_sv($csol->{$inst}, $c2idx);

    # map cluster_score_vector to sense_score_vector
    my @sense_sv = &vector_times_matrix(\@cluster_sv, $M);

    # Get sense with maximum score
    # @@ ties ?
    my $sscore_max = 0;
    my $sense_i = 0;
    my $sense_found = 0;
    my $i = 0;
    while($i<scalar(@sense_sv)) {
      if ($sense_sv[$i] > $sscore_max) {
	$sense_found = 1;
	$sense_i = $i;
	$sscore_max = $sense_sv[$i];
      }
      $i++;
    }
    # label instance with the sense of maximum score
    $ssol->{$inst} = $senses->[$sense_i] if $sense_found;
  }
}


# Create a hash { sense => cluster_score_vector }. If s and c are a sense
# and a cluster respectively, entry (s,c) is the score of cluster c in the
# sense s. Taking into account:
#
# 1. instances with more than one sense (in the gold standard)
#
# 2. The weights of the clusters (in the clustering solution) in each
#    instance


sub senses_score_vector {
  my ($instV, $csol, $gs, $cluster2idx, $ssv) = @_;

  foreach my $inst (@{$instV}) { # for each instance in the gs
    my $sdelta = 1 / scalar(@{$gs->{$inst}});
    foreach my $sense (@{$gs->{$inst}}) {
      my @inst_csv = &instance_cluster_sv($csol->{$inst}, $cluster2idx);
      # Multiply sense delta
      foreach my $aux (@inst_csv) {
	$aux*= $sdelta;
      }
      if (defined($ssv->{$sense})) {
	# Add inst_csv to sense
	&vector_add($ssv->{$sense}, \@inst_csv);
      } else {
	$ssv->{$sense} = \@inst_csv;
      }
    }
  }
}

# Create a (M x N) mapping matrix, where rows are clusters and columns are
# senses. If i and j are the i.th sense (s_i) and j.th cluster (c_j)
# respectively, entry (i,j) is the contitional probability P(s_i/c_j), that
# is, the probability of an instance of being of sense s_i, given that it is
# labeled with cluster c_j.

sub create_mapping_matrix {

  my ($ssv, $s2idx, $c2idx) = @_;

  my @senses = keys %{$s2idx}; # Auxiliary variable

  my $M = []; # the mapping matrix

  foreach my $cluster (keys %{$c2idx}) {
    my $j = $c2idx->{$cluster};
    my @sdist = map { 0 } @senses;
    foreach my $sense (@senses) {
      my $i = $s2idx->{$sense};
      $sdist[$i]+= $ssv->{$sense}->[$j];
    }
    &vector_normalize(\@sdist); # Make a probability distribution
    $M->[$j] = \@sdist;
  }
  return $M;
}

# Given a vector of clusters (with associated weigths) for an instance,
# return the resulting cluster score vector for this instance.

sub instance_cluster_sv {
  my ($clusters, $c2idx) = @_;

  my @sv = map { 0 } keys %{$c2idx};

  foreach my $cw (@{$clusters}) {
    my ($cluster, $weigth) = split(/\//, $cw);
    $weigth = 1 unless defined($weigth); # default weigth
    $sv[$c2idx->{$cluster}]+= $weigth;
  }
  return @sv;
}

# Get all the clusters occurring in a clustering solution.
#
# Return a vector with the clusters

sub get_clusters {

  my $cs = shift;

  my %clusters;

  foreach my $inst (keys(%{$cs})) {
    foreach my $aux (@{$cs->{$inst}}) {
      my $cluster = (split(/\//, $aux))[0]; # Skip weigth
      $clusters{$cluster} = 1;
    }
  }
  return sort keys %clusters;
}


# Get the senses together with its relative frequencies occurring in the
# gold standard. The frequencies are normalized.
#
# Return a hash { sense => freq }

sub get_gs_senses_freq {

  my $gs = shift;

  my %senses;
  my $N = 0;

  foreach my $inst (keys(%{$gs})) {
    my $delta = 1 / scalar(@{$gs->{$inst}});
    foreach my $s (@{$gs->{$inst}}) {
      $senses{$s}+= $delta;
    }
    $N++;
  }
  # Normalize
  foreach my $s (keys %senses) {
    $senses{$s}/= $N;
  }
  return %senses;
}

# Given a clustering solution (train+test) and the gold standard (train),
# return the test instances, i.e., those instances in clustering solution
# and not in gold standard

sub get_test_instances {

  my ($c_sol, $gs_sol) = @_;

  my @res;
  foreach my $inst (keys %{$c_sol}) {
    next if ($gs_sol->{$inst});
    push(@res, $inst);
  }
  return @res;
}

# Normalize a vector so it becomes a probability distribution

sub vector_normalize {

  my $vec = shift;

  my $tot = 0;
  my $i;
  foreach $i (@{$vec}) {
    $tot+=$i;
  }
  return unless ($tot);
  foreach $i (@{$vec}) {
    $i = $i / $tot;  # normalize
  }
}


# Add @v2 to @v1

sub vector_add {

  my ($v1, $v2) = @_;

  my $i=0;
  my $m = scalar(@{$v1}); 
  for (my $i=0; $i < $m; $i++) {
    $v1->[$i]+= $v2->[$i];
  }
}

# Multiply a (1 x m) vector v with a (m x n) matrix M
#
# Return a (1 x n) vector v*M

sub vector_times_matrix {

  my ($v, $M) = @_;

  my @res;

  my $m = scalar(@{$M});
  return () if ($m==0);
  die "vector_times_matrix: vector and matrix not compatible!\n" unless ($m == scalar(@{$v}));
  my $n = scalar(@{$M->[0]});

  my $i = 0;
  while ($i < $n) {
    my $aux = 0;
    my $j = 0;
    while ($j < $m) {
      $aux+= $v->[$j] * $M->[$j]->[$i];
      $j++;
    }
    push(@res, $aux);
    $i++;
  }
  return @res;
}


# Given a vector, return a hash { element => idx }

sub hash_idx {

  my $v = shift;

  my %H;
  my $idx = 0;
  my $m = scalar(@{$v});
  while($idx < $m) {
    $H{$v->[$idx]} = $idx;
    $idx++;
  }
  return %H;
}

# Write a word.c2s file with 4 elements:
# 1. N senses
# 2. (1xN) sense frequency vector (frequencies calculated over the train split)
# 3. M clusters
# 4. (MxN) mapping matrix from clusters to senses


sub write_cluster2sense {

  my ($out_dir, $word, $sense_freq, $senses, $clusters, $M) = @_;

  my $out_file = "$out_dir/$word.c2s";

  my $fh = IO::File->new(">$out_file");
  die "Can't create $out_file: $!\n" unless defined $fh;

  print $fh "Senses and frequencies:\n";
  print $fh join(" ", @{$senses});
  print $fh "\n";
  print $fh join(" ", map { $sense_freq->{$_} } @{$senses});
  print $fh "\n";
  print $fh "\n";
  print $fh "Clusters and mapping matrix:\n";
  print $fh join(" ", @{$clusters});
  print $fh "\n";
  foreach my $dist (@{$M}) {
    print $fh join(" ", @{$dist});
    print $fh "\n";
  }
  print $fh "\n";
}

sub discard_word {
  my ($word, $pos) = @_;

  return 1 if $pos =~ /^n/ && $word !~ /\.n/;
  return 1 if $pos =~ /^v/ && $word !~ /\.v/;
  return 0;
}

sub read_cluster_solution {

  my ($file, $CSolution) = @_;

  &read_key_file($file, $CSolution);

}

sub read_gold_standard {

  my ($file, $CSolution) = @_;

  my ($word, $instV);

  &read_key_file($file, $CSolution);
  # Expand senses
  while (($word, $instV) = each %{$CSolution}) {
    my ($inst, $cs);
    while (($inst, $cs) = each %{$instV}) {
      my @senses = split(/\#/, $cs->[0]);
      $cs = \@senses;
    }
  }
}

{

  my $line_number;

  sub read_key_file {

    my ($file, $CSolution) = @_;

    my $fh = IO::File->new($file);
    die "Can't open $file: $!\n" unless defined $fh;
    $line_number = 0;
    my $l;
    while ($l = &read_line_nospace($fh)) {
      chomp($l);
      my @C = split(/\s+/, $l);
      die "Input error in file $file,  line $line_number\nError: too few fields\n" if (scalar(@C) < 3);
      my $word = shift(@C);
      my $inst = shift(@C);
      &remove_comments(\@C);
      # @C now contains the clusters (or senses) assigned to this instance
      die "Input error in file $file,  line $line_number\nError: no labels\n" if (scalar(@C) == 0);
      if (!defined($CSolution->{$word})) {
	# first appearence of the word
	$CSolution->{$word} = {$inst => \@C};
      } else {
	$CSolution->{$word}->{$inst} = \@C;
      }
    }
  }

  sub remove_comments {

    my $v = shift;

    my $m = scalar(@{$v});
    my $i = 0;
    while ($i != $m) {
      if ($v->[$i] eq "!!") {
	splice(@{$v}, $i);
	last;
      }
      $i++;
    }
  }

  sub read_line_nospace {

    my $fh = shift;
    my $l;

    while($l = <$fh>) {
      $line_number++;
      last unless ($l =~ /^\s*$/o);
    }
    return $l;
  }
}

sub usage {
    my $str = shift;

  die <<"EOUSG"
  Error: $str\n
  Usage: create_supervised_keyfile.pl [-p all|n|v] [-O map_out_dir] clust_solution.key gold_standard_train.key
      -p filter by part of speech. 'n' takes only nouns into account, 'v' only
         verbs. 'all' doesn't filter anything. Default is 'all'
      -O Temporal directory for leaving the mapping matrix. If option is
         present, the program creates a file per word (extension .c2s) with
         the mapping matrix which maps clusters to senses

EOUSG
}
